"""
analyze_energy.py — Calibration analysis and confidence comparison (Assignment 2).

Produces:
    calibration_accuracy_vs_gap.png
    softmax_vs_energy_confidence.png

Usage:
    python src/analyze_energy.py \
        --baseline_ckpt runs/<softmax_run>/best.pt \
        --ebl_ckpt      runs/<ebm_run>/best.pt \
        --data_dir      data/idrid
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils   import set_seed, get_device
from dataset import get_dataloaders
from model   import build_model
from eval    import compute_energy_gap


def get_all_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).cpu()
            all_logits.append(logits)
            all_labels.extend(labels.numpy())
    return torch.cat(all_logits).numpy(), np.array(all_labels)


def plot_calibration_accuracy_vs_gap(gap, preds, labels, n_bins=10, save_path=None):
    """
    Binned calibration plot: accuracy as a function of energy gap.

    Interpretation: If EBL is well-calibrated, samples with larger energy
    gaps should be correct more often. A monotonically increasing curve
    suggests the energy gap is a reliable confidence proxy.
    """
    bins        = np.percentile(gap, np.linspace(0, 100, n_bins + 1))
    bin_accs    = []
    bin_centers = []
    bin_counts  = []

    for i in range(n_bins):
        mask = (gap >= bins[i]) & (gap < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        bin_accs.append(acc)
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_counts.append(mask.sum())

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(bin_centers, bin_accs, "o-", color="steelblue", label="Accuracy per bin")
    ax1.set_xlabel("Energy gap Δ(x) (binned)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=np.diff(bins).min() * 0.4,
            alpha=0.2, color="orange", label="Sample count")
    ax2.set_ylabel("Sample count", color="orange")

    ax1.set_title("Calibration: Accuracy vs Energy Gap (EBL)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Calibration → {save_path}")
    plt.close()


def plot_softmax_vs_energy_confidence(
    baseline_logits, baseline_labels, baseline_preds,
    ebl_logits, ebl_labels, ebl_preds,
    save_path=None
):
    """
    Side-by-side comparison:
    Left : softmax max-probability (baseline confidence proxy)
    Right: energy gap (EBL confidence proxy)

    Green = correct predictions, Red = incorrect.
    A good confidence measure should show green distributions
    shifted to the RIGHT (higher confidence).
    """
    # Baseline: softmax max probability
    baseline_probs      = torch.softmax(torch.tensor(baseline_logits), dim=1).numpy()
    softmax_confidence  = baseline_probs.max(axis=1)
    correct_base        = baseline_preds == baseline_labels
    wrong_base          = ~correct_base

    # EBL: energy gap
    ebl_energies  = -ebl_logits
    energy_gap    = compute_energy_gap(ebl_energies, ebl_preds)
    correct_ebl   = ebl_preds == ebl_labels
    wrong_ebl     = ~correct_ebl

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: softmax confidence
    bins_sm = np.linspace(0, 1, 30)
    axes[0].hist(softmax_confidence[correct_base], bins=bins_sm,
                 alpha=0.6, color="green",
                 label=f"Correct (n={correct_base.sum()})", density=True)
    axes[0].hist(softmax_confidence[wrong_base], bins=bins_sm,
                 alpha=0.6, color="red",
                 label=f"Wrong (n={wrong_base.sum()})", density=True)
    axes[0].set_xlabel("Softmax max probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Baseline: Softmax Confidence")
    axes[0].legend()

    # Right: energy gap
    bins_eg = np.linspace(energy_gap.min(), energy_gap.max(), 30)
    axes[1].hist(energy_gap[correct_ebl], bins=bins_eg,
                 alpha=0.6, color="green",
                 label=f"Correct (n={correct_ebl.sum()})", density=True)
    axes[1].hist(energy_gap[wrong_ebl], bins=bins_eg,
                 alpha=0.6, color="red",
                 label=f"Wrong (n={wrong_ebl.sum()})", density=True)
    axes[1].axvline(0, color="black", linestyle="--", label="gap = 0")
    axes[1].set_xlabel("Energy gap Δ(x)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("EBL: Energy Gap Confidence")
    axes[1].legend()

    plt.suptitle("Confidence Proxy: Softmax (Baseline) vs Energy Gap (EBL)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Confidence comparison → {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--ebl_ckpt",      required=True)
    parser.add_argument("--data_dir",      default="data/idrid")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--image_size",    type=int, default=224)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Load baseline
    base_ckpt    = torch.load(args.baseline_ckpt, map_location=device)
    class_names  = base_ckpt["class_names"]
    base_model   = build_model(len(class_names), pretrained=False).to(device)
    base_model.load_state_dict(base_ckpt["model_state"])

    # Load EBL
    ebl_ckpt     = torch.load(args.ebl_ckpt, map_location=device)
    ebl_model    = build_model(len(class_names), pretrained=False).to(device)
    ebl_model.load_state_dict(ebl_ckpt["model_state"])

    loaders = get_dataloaders(
        splits_csv=str(Path(args.data_dir) / "splits.csv"),
        image_dir=str(Path(args.data_dir) / "images"),
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    base_logits, base_labels = get_all_logits(base_model, loaders["test"], device)
    ebl_logits,  ebl_labels  = get_all_logits(ebl_model,  loaders["test"], device)

    base_preds = base_logits.argmax(axis=1)
    ebl_preds  = ebl_logits.argmax(axis=1)

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    ebl_energies = -ebl_logits
    gap          = compute_energy_gap(ebl_energies, ebl_preds)

    plot_calibration_accuracy_vs_gap(
        gap, ebl_preds, ebl_labels,
        save_path=out / "calibration_accuracy_vs_gap.png"
    )

    plot_softmax_vs_energy_confidence(
        base_logits, base_labels, base_preds,
        ebl_logits, ebl_labels, ebl_preds,
        save_path=out / "softmax_vs_energy_confidence.png"
    )
