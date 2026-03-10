"""
dataset.py — IDRiD data loading, stratified splitting, and augmentation.

Expected data layout:
    data/idrid/
        images/        <- all *.jpg or *.png files
        labels.csv     <- columns: image_id, label

Run standalone to verify your data and generate splits.csv:
    python src/dataset.py
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── ImageNet normalization (required when using pretrained ResNet) ──────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Returns the transform pipeline for a given split.

    IMPORTANT: Augmentation is applied ONLY to the training split.
    Val and test use only resize + crop + normalize so that evaluation
    is deterministic and fair.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.CenterCrop(image_size),          # removes black borders
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val or test — no augmentation
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class IDRiDDataset(Dataset):
    """
    PyTorch Dataset for IDRiD fundus images.

    Args:
        df          : DataFrame with columns [image_id, label]
        image_dir   : path to folder containing the images
        transform   : torchvision transform pipeline
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.classes   = sorted(df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["image_id"]

        # Try common extensions if exact filename not found
        if not img_path.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = self.image_dir / (row["image_id"] + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]
        return image, label


def create_splits(
    labels_csv: str,
    output_csv: str,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Create a stratified 70/15/15 train/val/test split and save splits.csv.

    Why stratified? IDRiD is class-imbalanced (grade 0 dominates).
    Without stratification, rare classes may vanish from val/test.

    Args:
        labels_csv  : path to the CSV with columns [image_id, label]
        output_csv  : where to write splits.csv
        seed        : random seed for reproducibility
        val_frac    : fraction of total data for validation
        test_frac   : fraction of total data for test

    Returns:
        DataFrame with columns [image_id, label, split]
    """
    df = pd.read_csv(labels_csv)
    assert {"image_id", "label"}.issubset(df.columns), \
        "labels.csv must have columns: image_id, label"

    # First split: train vs temp (val + test)
    temp_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(
        df, test_size=temp_frac, stratify=df["label"], random_state=seed
    )

    # Second split: val vs test (equal halves of temp)
    relative_test_frac = test_frac / temp_frac
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_frac, stratify=temp_df["label"],
        random_state=seed
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"

    splits = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    splits.to_csv(output_csv, index=False)

    # Print split summary
    for split_name in ["train", "val", "test"]:
        subset = splits[splits["split"] == split_name]
        print(f"  {split_name:5s}: {len(subset):4d} images | "
              f"class dist: {dict(subset['label'].value_counts().sort_index())}")

    return splits


def get_dataloaders(
    splits_csv: str,
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
) -> dict:
    """
    Build DataLoader objects for train, val, and test splits.

    Returns:
        dict with keys "train", "val", "test" mapping to DataLoaders,
        plus "num_classes" and "class_names".
    """
    splits = pd.read_csv(splits_csv)

    loaders    = {}
    class_names = sorted(splits["label"].unique().tolist())

    for split_name in ["train", "val", "test"]:
        subset    = splits[splits["split"] == split_name]
        transform = get_transforms(split_name, image_size)
        dataset   = IDRiDDataset(subset, image_dir, transform)
        shuffle   = (split_name == "train")
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    loaders["num_classes"] = len(class_names)
    loaders["class_names"] = class_names
    return loaders


def get_class_weights(splits_csv: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for the training split.

    WHY: IDRiD is heavily imbalanced. Without weighting, the baseline
    CrossEntropy model will learn to predict the majority class and
    achieve decent accuracy but terrible macro-F1. This makes the
    baseline look weaker than it is and creates an unfair comparison
    with EBL. Always use weighted CE for the baseline.

    Returns: Tensor of shape (num_classes,), ready to pass to
             nn.CrossEntropyLoss(weight=...)
    """
    splits = pd.read_csv(splits_csv)
    train  = splits[splits["split"] == "train"]
    counts = train["label"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum()  # normalize
    return torch.tensor(weights, dtype=torch.float32)


# ── Standalone verification ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",    default="data/idrid/labels.csv")
    parser.add_argument("--image_dir", default="data/idrid/images")
    parser.add_argument("--splits",    default="data/idrid/splits.csv")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    print("[dataset] Creating stratified splits...")
    create_splits(args.labels, args.splits, seed=args.seed)
    print(f"[dataset] splits.csv saved to {args.splits}")
    print("[dataset] Done. Run train.py when ready.")
