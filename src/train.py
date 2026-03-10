"""
train.py — Training script for both baseline and EBL experiments.

Usage:
    # Baseline (Assignment 1, Part C)
    python train.py --loss softmax --epochs 10

    # EBL sum-margin (Assignment 1, Part D)
    python train.py --loss ebm_margin --margin 1.0 --epochs 10

    # EBL hard-margin with regularization (Assignment 2)
    python train.py --loss ebm_margin_hard --margin 1.0 --lambda_reg 1e-3 --epochs 10
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils   import set_seed, get_device, save_metrics
from dataset import get_dataloaders, get_class_weights, create_splits
from model   import build_model
from losses  import build_loss
from eval    import run_evaluation   # reuse evaluation logic


def train_one_epoch(model, loader, criterion, optimizer, device, loss_type):
    model.train()
    total_loss    = 0.0
    active_fracs  = []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — essential for EBL losses which can produce
        # large gradients when many constraints are violated simultaneously.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Track active constraint fraction (EBL only)
        if hasattr(criterion, "active_fraction"):
            active_fracs.append(criterion.active_fraction(logits, labels))

    avg_loss   = total_loss / len(loader)
    avg_active = sum(active_fracs) / len(active_fracs) if active_fracs else None
    return avg_loss, avg_active


def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total += criterion(logits, labels).item()
    return total / len(loader)


def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()

    run_name = (
        f"{args.loss}_m{args.margin}_lam{args.lambda_reg}"
        f"_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
        f"_seed{args.seed}"
    )
    ckpt_dir = Path("runs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] {run_name}")

    # ── Data ───────────────────────────────────────────────────────────────
    splits_csv = Path(args.data_dir) / "splits.csv"
    if not splits_csv.exists():
        print("[data] splits.csv not found — creating from labels.csv ...")
        create_splits(
            labels_csv=str(Path(args.data_dir) / "labels.csv"),
            output_csv=str(splits_csv),
            seed=args.seed,
        )

    loaders     = get_dataloaders(
        splits_csv=str(splits_csv),
        image_dir=str(Path(args.data_dir) / "images"),
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    num_classes = loaders["num_classes"]
    class_names = loaders["class_names"]
    print(f"[data] {num_classes} classes: {class_names}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    class_weights = get_class_weights(str(splits_csv)).to(device)
    criterion = build_loss(
        loss_type=args.loss,
        class_weights=class_weights,   # ignored for EBL losses
        margin=args.margin,
        lambda_reg=args.lambda_reg,
    )

    # ── Optimizer + LR scheduler ───────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_macro_f1   = -1.0
    best_ckpt_path  = str(ckpt_dir / "best.pt")
    history         = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "active_frac": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, active_frac = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, args.loss
        )
        v_loss = val_loss(model, loaders["val"], criterion, device)

        # Evaluate macro-F1 on val (used for early stopping)
        val_metrics = run_evaluation(
            model, loaders["val"], device, class_names,
            output_dir=None,   # no plots during training
            split="val",
        )
        macro_f1 = val_metrics["macro_f1"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(v_loss)
        history["val_macro_f1"].append(macro_f1)
        history["active_frac"].append(active_frac)

        active_str = f"  active={active_frac:.2f}" if active_frac is not None else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={v_loss:.4f} | "
            f"val_macro_f1={macro_f1:.4f}{active_str}"
        )

        # Save best checkpoint
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "macro_f1":    macro_f1,
                "args":        vars(args),
                "class_names": class_names,
            }, best_ckpt_path)
            print(f"  ✓ New best val macro-F1 = {macro_f1:.4f} — checkpoint saved")

        # Red-flag check: if macro-F1 is stuck near chance after epoch 5,
        # something is wrong (data issue, missing class weights, etc.)
        if epoch == 5 and macro_f1 < (1.0 / num_classes + 0.05):
            print(
                f"\n⚠ WARNING: val macro-F1 = {macro_f1:.4f} after 5 epochs "
                f"(chance ≈ {1/num_classes:.2f}). "
                "Check: class weights, learning rate, data loading."
            )

    # Save training curves
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ───────────────────────────────────────────────
    print(f"\n[eval] Loading best checkpoint: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = run_evaluation(
        model, loaders["test"], device, class_names,
        output_dir="outputs",
        split="test",
        run_name=run_name,
        loss_type=args.loss,
    )

    # ── Save to metrics.csv ────────────────────────────────────────────────
    save_metrics("outputs/metrics.csv", {
        "seed":        args.seed,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "batch_size":  args.batch_size,
        "image_size":  args.image_size,
        "loss_type":   args.loss,
        "margin":      args.margin,
        "lambda_reg":  args.lambda_reg,
        "best_val_macro_f1": best_macro_f1,
        "test_acc":    test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    })

    print(f"\n[done] Test accuracy={test_metrics['accuracy']:.4f} | "
          f"macro-F1={test_metrics['macro_f1']:.4f}")
    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline or EBL model on IDRiD")

    # Data
    parser.add_argument("--data_dir",   default="data/idrid",
                        help="Path to IDRiD data folder (must contain labels.csv and images/)")
    parser.add_argument("--dataset",    default="idrid")

    # Loss
    parser.add_argument("--loss",       default="softmax",
                        choices=["softmax", "ebm_margin", "ebm_margin_hard"],
                        help="Loss function to use")
    parser.add_argument("--margin",     type=float, default=1.0,
                        help="EBL margin m (only used for EBL losses)")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                        help="L2 regularizer on logits (Assignment 2)")

    # Training
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--image_size", type=int,   default=224)
    parser.add_argument("--seed",       type=int,   default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)
    train(args)
