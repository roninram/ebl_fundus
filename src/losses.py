"""
losses.py — Cross-entropy baseline and Energy-Based Learning margin losses.

ENERGY CONVENTION (read this once carefully):
    logits  = f_theta(x)         shape: (B, K)
    energies = -logits           shape: (B, K)
    E(x, k) = -logits[:, k]

    Lower energy => model believes class k fits input x better.
    The correct class y should have the LOWEST energy after training.

TWO EBL LOSS VARIANTS:
    ebm_margin_sum  : original from LeCun et al., sums over all wrong classes
    ebm_margin_hard : uses only the hardest (closest) negative class

    The "hard" variant is more stable and is preferred in practice.
    Start with "sum" for Assignment 1, add "hard" in Assignment 2.

GRADIENT NOTE:
    Margin losses use max(0, ...) — when the constraint is already satisfied
    (value <= 0), the gradient is exactly zero ("dead constraint").
    Monitor active_fraction per batch. If it drops below ~5%, training
    has stalled — try reducing margin m or using a different variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Baseline ─────────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with optional class weights.

    Always use class_weights with IDRiD (it's imbalanced).
    Pass weights from dataset.get_class_weights() when building the loss.
    """

    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.ce(logits, targets)


# ── EBL: margin-sum loss ──────────────────────────────────────────────────────

class EBMMarginSumLoss(nn.Module):
    """
    Margin-based EBL loss — sums over ALL incorrect classes.
    From LeCun et al. 2006, Section 2.

        L(x, y) = sum_{k != y} max(0,  m  +  E(x,y)  -  E(x,k))
               = sum_{k != y} max(0,  m  -  logits[y]  +  logits[k])

    Intuition:
        For each wrong class k, we want E(x,k) > E(x,y) + m.
        Equivalently: logits[y] > logits[k] + m  (correct class wins by margin).
        If the constraint is violated, we incur a penalty proportional to
        how badly it is violated.

    Args:
        margin      : m, the minimum energy gap to enforce (default 1.0)
        lambda_reg  : L2 regularizer weight on logit magnitude (default 0)
                      Prevents energies from growing unbounded.
                      Sweep: {0, 1e-4, 1e-3, 1e-2} in Assignment 2.
    """

    def __init__(self, margin: float = 1.0, lambda_reg: float = 0.0):
        super().__init__()
        self.margin     = margin
        self.lambda_reg = lambda_reg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, K = logits.shape

        # Energy = -logit
        energies = -logits  # (B, K)

        # Energy of the correct class for each sample
        E_correct = energies[torch.arange(B), targets]  # (B,)

        # Broadcast for pairwise comparison: shape (B, K)
        E_correct_expanded = E_correct.unsqueeze(1).expand_as(energies)

        # Margin violation for each (sample, wrong_class) pair
        # violation[b, k] = max(0, m + E(x,y) - E(x,k))
        violations = torch.clamp(
            self.margin + E_correct_expanded - energies,
            min=0.0
        )  # (B, K)

        # Zero out the correct class (we only sum over k != y)
        mask = torch.ones(B, K, device=logits.device)
        mask[torch.arange(B), targets] = 0.0
        violations = violations * mask  # (B, K)

        loss = violations.sum(dim=1).mean()  # scalar

        # Optional: regularize logit magnitude to prevent energy explosion
        if self.lambda_reg > 0:
            reg = self.lambda_reg * (logits ** 2).sum(dim=1).mean()
            loss = loss + reg

        return loss

    def active_fraction(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Fraction of margin constraints that are currently violated (active).
        Log this per batch — if it drops < 5%, training may have stalled.
        """
        B, K = logits.shape
        energies          = -logits
        E_correct         = energies[torch.arange(B), targets].unsqueeze(1)
        violations        = self.margin + E_correct - energies
        mask              = torch.ones(B, K, device=logits.device)
        mask[torch.arange(B), targets] = 0.0
        active            = (violations > 0) * mask
        return active.sum().item() / (mask.sum().item() + 1e-8)


# ── EBL: margin-hard loss (Assignment 2) ─────────────────────────────────────

class EBMMarginHardLoss(nn.Module):
    """
    Margin-based EBL loss — uses only the HARDEST negative class.
    More stable than the sum variant; add in Assignment 2.

        L(x, y) = max(0,  m  +  E(x,y)  -  min_{k!=y} E(x,k))

    The "hardest negative" is the wrong class with the LOWEST energy
    (i.e., the one the model is most confused about).

    Args:
        margin     : m (default 1.0)
        lambda_reg : L2 regularizer on logits (default 0)
    """

    def __init__(self, margin: float = 1.0, lambda_reg: float = 0.0):
        super().__init__()
        self.margin     = margin
        self.lambda_reg = lambda_reg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, K = logits.shape
        energies = -logits

        E_correct = energies[torch.arange(B), targets]  # (B,)

        # Mask out correct class so we find min over wrong classes only
        energies_wrong           = energies.clone()
        energies_wrong[torch.arange(B), targets] = float("inf")
        E_hardest_negative, _    = energies_wrong.min(dim=1)  # (B,)

        violations = torch.clamp(
            self.margin + E_correct - E_hardest_negative,
            min=0.0
        )  # (B,)

        loss = violations.mean()

        if self.lambda_reg > 0:
            reg  = self.lambda_reg * (logits ** 2).sum(dim=1).mean()
            loss = loss + reg

        return loss

    def active_fraction(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        B, K             = logits.shape
        energies         = -logits
        E_correct        = energies[torch.arange(B), targets]
        ew               = energies.clone()
        ew[torch.arange(B), targets] = float("inf")
        E_hard, _        = ew.min(dim=1)
        violations       = self.margin + E_correct - E_hard
        return (violations > 0).float().mean().item()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_loss(
    loss_type: str,
    class_weights: torch.Tensor = None,
    margin: float = 1.0,
    lambda_reg: float = 0.0,
) -> nn.Module:
    """
    Loss factory.  Call this from train.py instead of instantiating directly.

    Args:
        loss_type     : "softmax" | "ebm_margin" | "ebm_margin_hard"
        class_weights : for softmax baseline (from dataset.get_class_weights)
        margin        : EBL margin m
        lambda_reg    : EBL L2 regularizer lambda

    Returns:
        A loss module with signature: loss(logits, targets) -> scalar
    """
    if loss_type == "softmax":
        return WeightedCrossEntropyLoss(class_weights)
    elif loss_type == "ebm_margin":
        return EBMMarginSumLoss(margin=margin, lambda_reg=lambda_reg)
    elif loss_type == "ebm_margin_hard":
        return EBMMarginHardLoss(margin=margin, lambda_reg=lambda_reg)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Choose: softmax | ebm_margin | ebm_margin_hard"
        )
