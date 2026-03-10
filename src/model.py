"""
model.py — ResNet-18 backbone with a configurable classification head.

The same model is used for both baseline (softmax CE) and EBL (margin loss).
The key insight: for EBL, the raw logits ARE the negated energies.
    E(x, k) = -f_theta(x)_k
So lower energy <=> higher logit <=> model prefers class k for input x.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet-18 with the final FC layer replaced.

    Args:
        num_classes : number of output classes (K)
        pretrained  : use ImageNet pretrained weights (strongly recommended)

    Returns:
        model with output shape (batch, num_classes)
        These output values are the LOGITS — used directly as -E(x,k) in EBL.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)

    return model


def get_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass returning raw logits (no softmax).
    Shape: (batch_size, num_classes)

    In EBL terms:
        logits[:, k] = f_theta(x)_k = -E(x, k)
    So to get energies:
        energies = -logits
    """
    return model(x)
