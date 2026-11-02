"""
SparseLayerBank: one sparse-code hippocampal bank for a single MLP layer.

The vector reduction of the ViT HippocampalBank: there are NO discrete slots.
A layer's output activation a_l (dim = layer out_features) is mapped through a
high-dimensional projection P and sparsified, producing a sparse code phi
(DG-style pattern separation). A single EMA vector b is the "superposed"
associative memory over all patterns seen — sparse codes of distinct inputs
are near-orthogonal, so they superpose with low crosstalk (cf. Kanerva SDM /
a sparse Hopfield trace). Retrieval is the cosine match s = cos(phi, b).

Pipeline per call:
    p   = project(a)                # (B, P)   P a_l  (fixed-random or learned)
    phi = sparsify(p)               # (B, P)   k-WTA (hard) or ReLU (L1-soft)
    s   = cos(phi, b) in [-1, 1]    # (B,)     per-sample match to the bank
    b  <- beta b + (1-beta) mean_b(phi)        EMA superposition update

Sparsity mechanisms (flag `sparsity`):
  - 'kwta': keep the top-k = round(sparsity_pct * P) activations per sample,
    zero the rest. Straight-through gradient (mask detached), so the kept
    units pass gradient to W in the step-3 consolidation write.
  - 'l1':   phi = ReLU(p); softly sparse, to be driven toward sparsity by an
    L1 penalty on the code during projection training (see l1_penalty()).
    Most meaningful with a learned projection.

Gradient note: project()/sparsify() are autograd-friendly (k-WTA uses a
straight-through mask) so step 3 can take d s / d W; the EMA update() runs
under no_grad and never touches the autograd graph.
"""

from typing import Optional

import torch
import torch.nn as nn


class SparseLayerBank(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        sparsity: str = "kwta",
        sparsity_pct: float = 0.05,
        proj_learned: bool = False,
        beta: float = 0.99,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if in_dim < 1 or proj_dim < 1:
            raise ValueError("in_dim/proj_dim must be >= 1")
        if sparsity not in ("kwta", "l1"):
            raise ValueError(f"unknown sparsity '{sparsity}' (kwta|l1)")
        if not (0.0 < sparsity_pct <= 1.0):
            raise ValueError("sparsity_pct must be in (0, 1]")

        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.sparsity = sparsity
        self.sparsity_pct = float(sparsity_pct)
        self.k = max(1, int(round(sparsity_pct * proj_dim)))
        self.beta = float(beta)
        self.eps = float(eps)
        self.proj_learned = bool(proj_learned)

        # Projection weight (proj_dim, in_dim), unit-norm rows. Built from a
        # DEDICATED generator so construction never perturbs the global RNG
        # stream (keeps the model byte-identical to stock sgd at init).
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        W0 = torch.randn(proj_dim, in_dim, generator=g)
        W0 = W0 / (W0.norm(dim=1, keepdim=True) + eps)

        if proj_learned:
            # nn.Linear's default init draws from the GLOBAL RNG; fork and
            # restore so a learned projection doesn't shift the stream either.
            rng_state = torch.random.get_rng_state()
            try:
                self.proj = nn.Linear(in_dim, proj_dim, bias=False)
            finally:
                torch.random.set_rng_state(rng_state)
            with torch.no_grad():
                self.proj.weight.copy_(W0)
        else:
            self.proj = None
            self.register_buffer("proj_weight", W0)

        # The single superposed memory vector and an update counter (so the
        # first update seeds b with the batch mean instead of crawling up
        # from zero through the EMA).
        self.register_buffer("bank", torch.zeros(proj_dim))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------
    # Code
    # ------------------------------------------------------------------

    def project(self, a: torch.Tensor) -> torch.Tensor:
        """a: (B, in_dim) -> p: (B, proj_dim)."""
        if self.proj is not None:
            return self.proj(a)
        return a @ self.proj_weight.t()

    def _kwta(self, p: torch.Tensor) -> torch.Tensor:
        """Keep the top-k activations per row, zero the rest. Straight-through:
        the boolean mask is detached, so kept units get gradient 1 and dropped
        units get 0."""
        k = min(self.k, p.shape[-1])
        # kth-largest value per row as the threshold.
        thresh = torch.topk(p, k, dim=-1).values[..., -1:].detach()
        mask = (p >= thresh).to(p.dtype)
        return p * mask

    def sparsify(self, p: torch.Tensor) -> torch.Tensor:
        if self.sparsity == "kwta":
            return self._kwta(p)
        # 'l1': soft, non-negative code; sparsity is encouraged by l1_penalty.
        return torch.relu(p)

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        """a (B, in_dim) -> sparse code phi (B, proj_dim)."""
        return self.sparsify(self.project(a))

    # ------------------------------------------------------------------
    # Retrieval and EMA update
    # ------------------------------------------------------------------

    def match(self, phi: torch.Tensor) -> torch.Tensor:
        """Per-sample cosine of the code against the bank: s in [-1, 1].
        Returns zeros while the bank is empty (n_updates == 0)."""
        b = self.bank
        num = phi @ b
        den = phi.norm(dim=-1).clamp_min(self.eps) * b.norm().clamp_min(self.eps)
        return num / den

    @torch.no_grad()
    def update(self, phi: torch.Tensor) -> None:
        """EMA-superpose the batch-mean code into the bank."""
        m = phi.detach().mean(dim=0)
        if int(self.n_updates) == 0:
            self.bank.copy_(m)
        else:
            self.bank.mul_(self.beta).add_(m, alpha=1.0 - self.beta)
        self.n_updates += 1

    def consolidation_grad(self, z_l: torch.Tensor,
                           gate: Optional[torch.Tensor] = None):
        """Gradient of the (gated) match w.r.t. the layer's pre-activation.

        Recomputes s_l = cos(phi(ReLU(z_l)), bank) through a fresh leaf and
        returns dL/dz_l for L = sum_b gate_b * s_b (the quantity we ASCEND to
        raise the match), plus the detached per-sample s_l.

        Caller turns this into the weight delta via the chain rule
        z_l = a_{l-1} W_l^T  =>  d L / d W_l = G_z^T @ a_{l-1}.

        gate: per-sample weights (the previous layer's match s_{l-1} in the
        cross-layer-gated variant); None means ungated (all ones, s_0 = 1).
        k-WTA's straight-through mask lets gradient reach the kept units.
        """
        with torch.enable_grad():
            z = z_l.detach().requires_grad_(True)
            phi = self.sparsify(self.project(torch.relu(z)))
            s = self.match(phi)                          # (B,)
            total = s.sum() if gate is None else (gate.detach() * s).sum()
            (G_z,) = torch.autograd.grad(total, z)
        return G_z, s.detach()

    def l1_penalty(self, phi: torch.Tensor) -> torch.Tensor:
        """Mean L1 of the code — the sparsity-encouraging term for 'l1' mode
        projection training (wired in a later step)."""
        return phi.abs().mean()

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, proj_dim={self.proj_dim}, "
            f"sparsity={self.sparsity}@{self.sparsity_pct} (k={self.k}), "
            f"proj_learned={self.proj_learned}, beta={self.beta}"
        )
