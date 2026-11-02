"""SlotFeatureDecoder: maps a bank slot's stored value content to a
synthetic CLS feature for slot-prioritized replay (head-mode).

The decoder's *input* is a deterministic summary of a slot's already-
stored value tensors (see ConsolidationModule.slot_value_matrix) — NOT a
per-slot learned embedding. That distinction matters: a learned per-slot
embedding would be feature-storage in disguise (it would let the model
memorize one feature per slot), violating the no-raw-storage constraint.
Decoding from the bank's own content keeps the only learnable state a
small, slot-agnostic MLP.

Optional Gaussian noise on the hidden activation lets one slot generate a
*distribution* of synthetic features across many replays (variety),
rather than a single point. Noise is off by default and is meant to be
enabled for replay generation while kept off for the anchoring MSE, whose
target is the real (mean) feature.
"""

import torch
import torch.nn as nn


class SlotFeatureDecoder(nn.Module):
    def __init__(
        self,
        val_dim: int,
        feature_dim: int,
        hidden: int = 384,
        noise: float = 0.0,
    ):
        """
        Args:
            val_dim: flattened per-slot value-summary dimension (the
                decoder's input width).
            feature_dim: output width = the model's pre-head CLS feature
                dim (D, e.g. 384 for ViT-S/16).
            hidden: MLP hidden width.
            noise: std of Gaussian noise added to the hidden activation
                when forward(add_noise=True). 0 disables.
        """
        super().__init__()
        if val_dim < 1 or feature_dim < 1 or hidden < 1:
            raise ValueError("val_dim/feature_dim/hidden must be >= 1")
        if noise < 0:
            raise ValueError(f"noise must be >= 0, got {noise}")
        self.fc1 = nn.Linear(val_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, feature_dim)
        self.noise = float(noise)

    def forward(self, slot_values: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """slot_values: (B, val_dim) -> (B, feature_dim).

        With add_noise=True and noise>0, injects Gaussian noise on the
        hidden activation so repeated calls on the same slot value produce
        varied features (reparameterization-style stochasticity).
        """
        h = self.act(self.fc1(slot_values))
        if add_noise and self.noise > 0:
            h = h + self.noise * torch.randn_like(h)
        return self.fc2(h)
