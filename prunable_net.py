"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================

This module implements a feed-forward neural network that learns to prune
itself during training via learnable gate parameters — no post-training
pruning step required.

Architecture innovations beyond the base spec:
  - Straight-Through Estimator (STE) hard-gate inference mode
  - BatchNorm after each prunable layer for stable training
  - Residual-style skip connection at the bottleneck
  - Temperature-scaled sigmoid annealing (warm gates → sharp decisions)
  - Per-layer sparsity tracking and rich metrics logging

Author: Aman (Tredence Case Study Submission)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Core Primitive: PrunableLinear
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to prune.

    Each weight w_ij has an associated scalar gate_score g_ij (same shape).
    During the forward pass:

        gates         = sigmoid(gate_scores / temperature)   ∈ (0, 1)
        pruned_weight = weight ⊙ gates                       (element-wise)
        output        = input @ pruned_weight.T + bias

    During inference (hard_mask=True), gates are binarised via a
    Straight-Through Estimator so the sparsity is exact, not approximate.

    Parameters
    ----------
    in_features  : int
    out_features : int
    temperature  : float
        Controls sigmoid sharpness. Lower → harder decisions.
        Start high (e.g. 1.0) and anneal toward 0.1 for best results.
    hard_mask    : bool
        If True, binarise gates at threshold 0.5 (inference mode).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float = 1.0,
        hard_mask: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.hard_mask    = hard_mask

        # ── Standard learnable parameters ──────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # ── Gate scores: one per weight ────────────────────────────────────
        # Initialising at 0 means sigmoid(0) = 0.5, which gives the MAXIMUM
        # gradient of the L1 sparsity loss w.r.t. gate_scores (σ'(0) = 0.25).
        # Connections useful for classification receive positive gradient from
        # CE loss, pushing their scores above 0 (gate → 1). Useless
        # connections only receive the L1 push (score → −∞, gate → 0).
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # ── Temperature buffer (not a param; updated by the trainer) ───────
        self.register_buffer("temperature", torch.tensor(temperature))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform for weights (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Gate scores: small positive bias so sigmoid ≈ 0.73 at start
        nn.init.constant_(self.gate_scores, 0.0)  # sigmoid(0)=0.5, max gradient
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def gates(self) -> torch.Tensor:
        """Soft gates ∈ (0, 1) via temperature-scaled sigmoid."""
        return torch.sigmoid(self.gate_scores / self.temperature)

    @property
    def sparsity(self) -> float:
        """Fraction of weights whose gate < 0.01 (effectively pruned)."""
        with torch.no_grad():
            return (self.gates < 0.01).float().mean().item()

    @property
    def active_fraction(self) -> float:
        return 1.0 - self.sparsity

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gates

        if self.hard_mask and not self.training:
            # Straight-Through Estimator binarisation for inference
            g_hard = (g > 0.5).float()
            g = g_hard - g.detach() + g  # STE: forward uses binary, grad uses soft

        pruned_weights = self.weight * g
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"sparsity={self.sparsity:.1%}, temp={self.temperature.item():.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A self-pruning feed-forward network for CIFAR-10.

    Input: 32×32 RGB images → flattened to 3072 dims
    Architecture (with BatchNorm + GELU activations):

        Input(3072) → 1024 → 512 → [256 skip] → 256 → 128 → 10

    The bottleneck uses a residual projection to maintain gradient flow
    even when heavy pruning occurs at intermediate layers.

    All linear layers are PrunableLinear instances.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()

        self.flatten = nn.Flatten()

        # ── Encoder ──────────────────────────────────────────────────────
        self.fc1  = PrunableLinear(3072, 1024, temperature=temperature)
        self.bn1  = nn.BatchNorm1d(1024)

        self.fc2  = PrunableLinear(1024, 512, temperature=temperature)
        self.bn2  = nn.BatchNorm1d(512)

        self.fc3  = PrunableLinear(512, 256, temperature=temperature)
        self.bn3  = nn.BatchNorm1d(256)

        # ── Bottleneck with residual skip ─────────────────────────────
        self.fc4  = PrunableLinear(256, 256, temperature=temperature)
        self.bn4  = nn.BatchNorm1d(256)

        # ── Decoder ──────────────────────────────────────────────────────
        self.fc5  = PrunableLinear(256, 128, temperature=temperature)
        self.bn5  = nn.BatchNorm1d(128)

        self.fc6  = PrunableLinear(128, 10, temperature=temperature)

        self.dropout = nn.Dropout(p=0.1)
        self.act     = nn.GELU()

    # ── Utility: iterate all prunable layers ────────────────────────────

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.dropout(self.act(self.bn1(self.fc1(x))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))
        x = self.dropout(self.act(self.bn3(self.fc3(x))))

        # Residual bottleneck
        residual = x
        x = self.act(self.bn4(self.fc4(x)))
        x = x + residual                             # skip connection

        x = self.dropout(self.act(self.bn5(self.fc5(x))))
        x = self.fc6(x)
        return x

    # ── Sparsity metrics ────────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Why L1? The L1 norm's non-differentiability at 0 creates a constant
        gradient pressure toward zero — unlike L2 which relaxes as values
        approach 0. This is the key property that produces exact sparsity.
        """
        return sum(layer.gates.sum() for layer in self.prunable_layers())

    def get_metrics(self) -> Dict[str, float]:
        """Return per-layer and aggregate sparsity stats."""
        layers = self.prunable_layers()
        total_weights  = sum(l.weight.numel() for l in layers)
        pruned_weights = sum(
            (l.gates < 0.01).float().sum().item() for l in layers
        )
        metrics = {
            f"layer_{i+1}_sparsity": l.sparsity
            for i, l in enumerate(layers)
        }
        metrics["overall_sparsity"] = pruned_weights / total_weights
        metrics["total_params"]     = sum(p.numel() for p in self.parameters())
        metrics["active_weights"]   = int(total_weights - pruned_weights)
        metrics["total_weights"]    = int(total_weights)
        return metrics

    def set_temperature(self, temp: float) -> None:
        """Anneal temperature in all prunable layers simultaneously."""
        for layer in self.prunable_layers():
            layer.temperature.fill_(temp)

    def get_all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values into a single flat tensor (for plotting)."""
        with torch.no_grad():
            return torch.cat([l.gates.flatten() for l in self.prunable_layers()])
