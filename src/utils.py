"""
utils.py — Reproducibility helpers and shared utilities.

IMPORTANT: Call set_seed() at the very top of every script.
All three libraries (torch, numpy, random) must be seeded or
results will differ across runs even with the same --seed flag.
"""

import random
import numpy as np
import torch
import csv
import os
from pathlib import Path


def set_seed(seed: int = 42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] Set global seed = {seed}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    return device


def save_metrics(path: str, row: dict):
    """
    Append one row to metrics.csv.
    Creates the file with headers if it doesn't exist.
    
    Expected keys: seed, epochs, lr, batch_size, image_size,
                   loss_type, margin, lambda_reg, test_acc,
                   test_macro_f1, notes
    """
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[metrics] Saved to {path}")


def count_active_constraints(loss_tensor: torch.Tensor) -> float:
    """
    Monitor what fraction of margin constraints are 'active' (>0).
    A very low fraction means the model has already satisfied most
    margins and gradients will be sparse — a useful diagnostic.
    """
    return (loss_tensor > 0).float().mean().item()
