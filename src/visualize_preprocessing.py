"""
visualize_preprocessing.py — Show 4 raw + 4 processed fundus images.

Required for the Assignment 1 report (Part B3).
Run after dataset.py has confirmed your data layout is correct.

Usage:
    python src/visualize_preprocessing.py --data_dir data/idrid --n 4
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_transforms


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a displayable (H, W, 3) uint8 array."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def main(args):
    data_dir  = Path(args.data_dir)
    labels_df = pd.read_csv(data_dir / "labels.csv")
    image_dir = data_dir / "images"
    transform = get_transforms("train", image_size=args.image_size)

    # Sample n images
    samples = labels_df.sample(n=args.n, random_state=42)
    fig, axes = plt.subplots(2, args.n, figsize=(4 * args.n, 8))

    for col, (_, row) in enumerate(samples.iterrows()):
        img_id = row["image_id"]
        label  = row["label"]

        # Find file
        img_path = image_dir / img_id
        if not img_path.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = image_dir / (img_id + ext)
                if cand.exists():
                    img_path = cand
                    break

        raw = Image.open(img_path).convert("RGB")
        processed_tensor = transform(raw)
        processed = unnormalize(processed_tensor)

        # Top row: raw
        axes[0, col].imshow(raw)
        axes[0, col].set_title(f"Raw\n{img_id[:12]}…\nlabel={label}", fontsize=8)
        axes[0, col].axis("off")

        # Bottom row: processed
        axes[1, col].imshow(processed)
        axes[1, col].set_title(f"Processed\n224×224", fontsize=8)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Raw", fontsize=11, labelpad=30)
    axes[1, 0].set_ylabel("Processed", fontsize=11, labelpad=30)

    plt.suptitle("Fundus Image Preprocessing: Raw vs Processed", fontsize=13)
    plt.tight_layout()
    out = Path("outputs") / "preprocessing_examples.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/idrid")
    parser.add_argument("--n",          type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
