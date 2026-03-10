"""
eval.py — Evaluation, plotting, and energy diagnostics.

Produces:
    confusion_matrix.png
    loss_curve.png           (when called from train.py)
    energy_gap_hist.png      (EBL only)
    energy_gap_correct_vs_wrong.png  (EBL only)

Usage:
    python eval.py --ckpt runs/<run_name>/best.pt --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils   import set_seed, get_device
from dataset import get_dataloaders
from model   import build_model


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    model,
    loader,
    device,
    class_names: list,
    output_dir: str = None,
    split: str = "test",
    run_name: str = "",
    loss_type: str = "softmax",
) -> dict:
    """
    Run inference, compute metrics, and optionally save plots.

    Returns dict with: accuracy, macro_f1, per_class (dict)
    """
    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds, target_names=class_names,
        zero_division=0, output_dict=True
    )

    print(f"\n[{split}] Accuracy = {accuracy:.4f} | Macro-F1 = {macro_f1:.4f}")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        prefix = f"{run_name}_{split}" if run_name else split

        # Confusion matrix
        plot_confusion_matrix(
            all_labels, all_preds, class_names,
            save_path=output_dir / f"{prefix}_confusion_matrix.png"
        )

        # Energy plots (EBL only)
        if loss_type in ("ebm_margin", "ebm_margin_hard"):
            energies   = -all_logits
            energy_gap = compute_energy_gap(energies, all_preds)
            plot_energy_gap_hist(
                energy_gap,
                save_path=output_dir / f"{prefix}_energy_gap_hist.png"
            )
            plot_energy_gap_correct_vs_wrong(
                energy_gap, all_preds, all_labels,
                save_path=output_dir / f"{prefix}_energy_gap_correct_vs_wrong.png"
            )

    return {
        "accuracy":   accuracy,
        "macro_f1":   macro_f1,
        "per_class":  report,
        "logits":     all_logits,
        "preds":      all_preds,
        "labels":     all_labels,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Confusion matrix → {save_path}")


def plot_loss_curve(history: dict, save_path: str):
    """
    Plot training and validation loss curves.
    Call after training with the history dict from train.py.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()

    axes[1].plot(epochs, history["val_macro_f1"], color="green", label="Val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].legend()

    # Overlay active constraint fraction for EBL
    if any(v is not None for v in history.get("active_frac", [])):
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["active_frac"], "--", color="orange", alpha=0.6,
                 label="Active frac")
        ax2.set_ylabel("Active constraint fraction", color="orange")
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Loss curve → {save_path}")


# ── Energy diagnostics ────────────────────────────────────────────────────────

def compute_energy_gap(energies: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Compute the energy gap for each sample:
        Δ(x) = min_{k ≠ ŷ} E(x, k)  -  E(x, ŷ)

    A LARGE positive gap means the model is confident (the predicted class
    has much lower energy than all alternatives).
    A gap near zero or negative indicates uncertainty or an error.

    Args:
        energies : (N, K) array of energies (-logits)
        preds    : (N,)   predicted class indices

    Returns:
        gap : (N,) array of energy gaps
    """
    N, K = energies.shape
    gap  = np.zeros(N)
    for i in range(N):
        E_pred = energies[i, preds[i]]
        others = [energies[i, k] for k in range(K) if k != preds[i]]
        E_min_other = min(others)
        gap[i] = E_min_other - E_pred   # positive = confident
    return gap


def plot_energy_gap_hist(gap: np.ndarray, save_path):
    plt.figure(figsize=(7, 4))
    plt.hist(gap, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(0, color="red", linestyle="--", label="gap = 0 (decision boundary)")
    plt.xlabel("Energy gap Δ(x)")
    plt.ylabel("Count")
    plt.title("Energy Gap Distribution (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Energy gap histogram → {save_path}")


def plot_energy_gap_correct_vs_wrong(gap, preds, labels, save_path):
    correct = gap[preds == labels]
    wrong   = gap[preds != labels]

    plt.figure(figsize=(8, 4))
    bins = np.linspace(gap.min(), gap.max(), 40)
    plt.hist(correct, bins=bins, alpha=0.6, color="green",
             label=f"Correct (n={len(correct)})", density=True)
    plt.hist(wrong,   bins=bins, alpha=0.6, color="red",
             label=f"Wrong   (n={len(wrong)})", density=True)
    plt.axvline(0, color="black", linestyle="--", label="gap = 0")
    plt.xlabel("Energy gap Δ(x)")
    plt.ylabel("Density")
    plt.title("Energy Gap: Correct vs Incorrect Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Energy gap correct vs wrong → {save_path}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--ckpt",       required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data_dir",   default="data/idrid")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    ckpt        = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    train_args  = ckpt["args"]

    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    loaders = get_dataloaders(
        splits_csv=str(Path(args.data_dir) / "splits.csv"),
        image_dir=str(Path(args.data_dir) / "images"),
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    run_name   = Path(args.ckpt).parent.name
    loss_type  = train_args.get("loss", "softmax")

    metrics = run_evaluation(
        model, loaders[args.split], device, class_names,
        output_dir="outputs",
        split=args.split,
        run_name=run_name,
        loss_type=loss_type,
    )

    # Save loss curves if history.json exists
    history_path = Path(args.ckpt).parent / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curve(history, save_path=f"outputs/{run_name}_loss_curve.png")
