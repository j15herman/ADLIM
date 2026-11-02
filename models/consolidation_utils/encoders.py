"""
Encoders for the hippocampal-indexed consolidation module.

CovarianceKeyEncoder
    Legacy. Encodes X into a flat D² covariance key. Retained for back-
    compat / ablations. Collapsed bank slots at ViT-S/16's D=384 even
    after centering and kWTA.

SlicedWassersteinKeyEncoder
    Legacy. SW signature over the per-sample token distribution as a
    bank key. Lower-dim (K*M=1024) than cov; matched cov on probe metrics
    but did not exceed naive.

SlicedWassersteinEncoder
    Encodes Y into a quantile signature (M, K) per head. Used for the
    output-side bank values; gradient flows back through it. UNCHANGED.

ChunkedRawKeyEncoder
    Current key encoder. Splits the (padded) input sequence into n_chunks
    contiguous chunks along the token axis and emits each chunk as a flat
    chunk_length*D vector. Comparison happens at the bank: position-aligned
    per-chunk cosine, aggregated over chunks. This preserves the most
    information about X for pattern separation (no compression to second
    moments or quantile shape) and remains permutation-aware within the
    chunked positional structure that ViT actually has.
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Legacy encoders. Kept for reproducibility of prior runs / ablations.
# =====================================================================


class CovarianceKeyEncoder(nn.Module):
    """
    Args:
        input_dim: D, the feature dimension of X.
        apply_ln: if True, LayerNorm is applied to X before computing the
            covariance.
        center_running_mean: if True, maintain a running EMA of the mean
            covariance key and subtract it before returning.
        center_momentum: EMA momentum for the running mean. Default 0.99.
        kwta_sparsity: if > 0 and < 1, apply k-winners-take-all to the key
            after centering. 0 disables.
        eps: LayerNorm epsilon (only used when apply_ln=True).
    """

    def __init__(
        self,
        input_dim: int,
        apply_ln: bool = False,
        center_running_mean: bool = True,
        center_momentum: float = 0.99,
        kwta_sparsity: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not 0.0 < center_momentum < 1.0:
            raise ValueError(
                f"center_momentum must be in (0,1), got {center_momentum}"
            )
        if not 0.0 <= kwta_sparsity < 1.0:
            raise ValueError(
                f"kwta_sparsity must be in [0,1), got {kwta_sparsity}"
            )
        self.input_dim = input_dim
        self.apply_ln = apply_ln
        self.center_running_mean = center_running_mean
        self.center_momentum = center_momentum
        self.kwta_sparsity = kwta_sparsity
        self.kwta_k = (
            int(math.ceil(kwta_sparsity * input_dim * input_dim))
            if kwta_sparsity > 0 else 0
        )
        if apply_ln:
            self.ln = nn.LayerNorm(input_dim, eps=eps)
        if center_running_mean:
            self.register_buffer(
                "running_mean", torch.zeros(input_dim * input_dim)
            )
            self.register_buffer(
                "running_var", torch.ones(input_dim * input_dim)
            )
            self.register_buffer(
                "mean_initialized", torch.zeros((), dtype=torch.bool)
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        if X.dim() != 3:
            raise ValueError(
                f"Expected X of rank 2 or 3, got rank {X.dim()} (shape {tuple(X.shape)})"
            )
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {X.shape[-1]}"
            )

        if self.apply_ln:
            X = self.ln(X)

        N = X.shape[1]
        C = torch.einsum("bnd,bne->bde", X, X) / N
        key = C.flatten(start_dim=1)

        if self.center_running_mean:
            if self.training:
                batch_mean = key.detach().mean(dim=0)
                with torch.no_grad():
                    if not bool(self.mean_initialized):
                        self.running_mean.copy_(batch_mean)
                        self.mean_initialized.fill_(True)
                    else:
                        m = self.center_momentum
                        self.running_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            key = key - self.running_mean

            if self.training:
                batch_var = key.detach().pow(2).mean(dim=0)
                with torch.no_grad():
                    m = self.center_momentum
                    self.running_var.mul_(m).add_(batch_var, alpha=1 - m)

        if self.kwta_k > 0:
            if self.center_running_mean:
                std = (self.running_var + 1e-4).sqrt()
                ranked_key = key / std
            else:
                ranked_key = key
            abs_ranked = ranked_key.abs()
            _, top_idx = abs_ranked.topk(self.kwta_k, dim=-1)
            mask = torch.zeros_like(key, dtype=torch.bool)
            mask.scatter_(-1, top_idx, True)
            key = key * mask

        if single:
            key = key.squeeze(0)
        return key


class SlicedWassersteinKeyEncoder(nn.Module):
    """
    Legacy SW-signature key encoder over per-sample token distribution.
    Kept for back-compat / ablation. See ChunkedRawKeyEncoder for current.
    """

    def __init__(
        self,
        input_dim: int,
        n_directions: int = 32,
        n_quantiles: int = 32,
        center_running_mean: bool = True,
        center_momentum: float = 0.99,
        standardize: bool = True,
        seed: Optional[int] = None,
        eps: float = 1e-4,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {n_directions}")
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
        if not 0.0 < center_momentum < 1.0:
            raise ValueError(
                f"center_momentum must be in (0,1), got {center_momentum}"
            )

        self.input_dim = input_dim
        self.K = n_directions
        self.M = n_quantiles
        self.center_running_mean = center_running_mean
        self.center_momentum = center_momentum
        self.standardize = standardize
        self.eps = eps
        self.key_dim = n_directions * n_quantiles

        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            U = torch.randn(input_dim, n_directions, generator=gen)
        else:
            U = torch.randn(input_dim, n_directions)
        U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("U", U)

        if center_running_mean:
            self.register_buffer("running_mean", torch.zeros(self.key_dim))
            self.register_buffer("running_var", torch.ones(self.key_dim))
            self.register_buffer(
                "mean_initialized", torch.zeros((), dtype=torch.bool)
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        if X.dim() != 3:
            raise ValueError(f"Expected X of rank 2 or 3, got rank {X.dim()}")
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {X.shape[-1]}"
            )

        B, N, _ = X.shape
        if N < 2:
            raise ValueError(f"Need at least 2 tokens, got N={N}")

        Z = X @ self.U
        Z_sorted, _ = Z.sort(dim=1)

        positions = (
            (torch.arange(self.M, dtype=X.dtype, device=X.device) + 0.5) * N / self.M
            - 0.5
        )
        positions = positions.clamp(0.0, float(N - 1))
        idx_lo = positions.floor().long()
        idx_hi = (idx_lo + 1).clamp_max(N - 1)
        w = (positions - idx_lo.to(X.dtype)).view(1, self.M, 1)

        Z_lo = Z_sorted[:, idx_lo, :]
        Z_hi = Z_sorted[:, idx_hi, :]
        S = (1.0 - w) * Z_lo + w * Z_hi

        key = S.reshape(B, self.M * self.K)

        if self.center_running_mean:
            if self.training:
                batch_mean = key.detach().mean(dim=0)
                with torch.no_grad():
                    if not bool(self.mean_initialized):
                        self.running_mean.copy_(batch_mean)
                        self.mean_initialized.fill_(True)
                    else:
                        m = self.center_momentum
                        self.running_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            key = key - self.running_mean

            if self.training:
                batch_var = key.detach().pow(2).mean(dim=0)
                with torch.no_grad():
                    m = self.center_momentum
                    self.running_var.mul_(m).add_(batch_var, alpha=1 - m)

            if self.standardize:
                std = (self.running_var + self.eps).sqrt()
                key = key / std

        if single:
            key = key.squeeze(0)
        return key


# =====================================================================
# Output-side encoder. Unchanged.
# =====================================================================


class SlicedWassersteinEncoder(nn.Module):
    """
    Output-side SW encoder. Differentiable in Y; backprop through it
    drives the weight update.

    Args:
        out_dim: d_out, the dimension of Y's last axis (e.g. d_head for
            per-head attention consolidation).
        n_directions: K, number of random projection directions.
        n_quantiles: M, number of quantile samples per direction.
        seed: optional seed for sampling random directions.
    """

    def __init__(
        self,
        out_dim: int,
        n_directions: int = 32,
        n_quantiles: int = 32,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {n_directions}")
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

        self.out_dim = out_dim
        self.K = n_directions
        self.M = n_quantiles

        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            U = torch.randn(out_dim, n_directions, generator=gen)
        else:
            U = torch.randn(out_dim, n_directions)
        U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("U", U)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        single = Y.dim() == 2
        if single:
            Y = Y.unsqueeze(0)
        if Y.dim() != 3:
            raise ValueError(f"Expected Y of rank 2 or 3, got rank {Y.dim()}")
        if Y.shape[-1] != self.out_dim:
            raise ValueError(
                f"Expected last dim {self.out_dim}, got {Y.shape[-1]}"
            )

        B, N, _ = Y.shape
        if N < 2:
            raise ValueError(f"Need at least 2 tokens, got N={N}")

        Z = Y @ self.U
        Z_sorted, _ = Z.sort(dim=1)

        positions = (
            (torch.arange(self.M, dtype=Y.dtype, device=Y.device) + 0.5) * N / self.M
            - 0.5
        )
        positions = positions.clamp(0.0, float(N - 1))
        idx_lo = positions.floor().long()
        idx_hi = (idx_lo + 1).clamp_max(N - 1)
        w = (positions - idx_lo.to(Y.dtype)).view(1, self.M, 1)

        Z_lo = Z_sorted[:, idx_lo, :]
        Z_hi = Z_sorted[:, idx_hi, :]
        S = (1.0 - w) * Z_lo + w * Z_hi

        if single:
            S = S.squeeze(0)
        return S


# =====================================================================
# Current key encoder: chunked raw tokens.
# =====================================================================


class ChunkedRawKeyEncoder(nn.Module):
    """
    Pad-and-chunk the input sequence, return per-chunk flat vectors plus
    a chunk-level pad mask. The bank consumes these to do position-aligned
    per-chunk cosine.

    Args:
        input_dim: D, feature dim of X.
        max_seq_length: N_max, the length to pad every sample up to. Must
            be divisible by chunk_length. For ViT-S/16 at 224x224, this is
            197 (1 CLS + 14*14 patches).
        chunk_length: L, tokens per chunk. With L = N_max, n_chunks=1 and
            each sample contributes a single full-sequence vector.
        center: maintain per-feature running mean and subtract from tokens
            before chunking. Default True.
        standardize: divide by per-feature running std after centering.
            Default True.
        center_momentum: EMA momentum for running stats. Default 0.99.
        eps: floor for std division.

    Forward:
        X: (B, N, D), N <= max_seq_length. Will be right-padded to N_max.
        seq_lengths: optional (B,) long tensor of true per-sample lengths.
            If None, assumes every sample is fully filled (no padding).

    Returns:
        chunks_flat: (B, n_chunks, chunk_length * D), centered & standardized
            (if requested), with pad-token positions zeroed so they
            contribute nothing to downstream dot products.
        pad_mask: (B, n_chunks) bool, True iff the entire chunk falls beyond
            seq_lengths[b]. Partial-pad chunks are flagged False (treated as
            real; their pad tokens are zero so they contribute zero to the
            chunk vector).
    """

    def __init__(
        self,
        input_dim: int,
        max_seq_length: int,
        chunk_length: int,
        center: bool = True,
        standardize: bool = True,
        center_momentum: float = 0.99,
        eps: float = 1e-4,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {max_seq_length}")
        if chunk_length < 1:
            raise ValueError(f"chunk_length must be >= 1, got {chunk_length}")
        if max_seq_length % chunk_length != 0:
            raise ValueError(
                f"max_seq_length ({max_seq_length}) must be divisible by "
                f"chunk_length ({chunk_length})"
            )
        if not 0.0 < center_momentum < 1.0:
            raise ValueError(
                f"center_momentum must be in (0,1), got {center_momentum}"
            )

        self.input_dim = input_dim
        self.max_seq_length = max_seq_length
        self.chunk_length = chunk_length
        self.n_chunks = max_seq_length // chunk_length
        self.chunk_dim = chunk_length * input_dim
        self.center = center
        self.standardize = standardize
        self.center_momentum = center_momentum
        self.eps = eps

        if center:
            self.register_buffer("running_mean", torch.zeros(input_dim))
            self.register_buffer("running_var", torch.ones(input_dim))
            self.register_buffer(
                "stats_initialized", torch.zeros((), dtype=torch.bool)
            )

    def forward(
        self,
        X: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        if X.dim() != 3:
            raise ValueError(
                f"Expected X of rank 2 or 3, got rank {X.dim()} (shape {tuple(X.shape)})"
            )
        B, N, D = X.shape
        if D != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {D}"
            )
        if N > self.max_seq_length:
            raise ValueError(
                f"Input sequence length {N} exceeds max_seq_length "
                f"{self.max_seq_length}"
            )

        # Right-pad to max_seq_length with zeros if needed.
        if N < self.max_seq_length:
            X = F.pad(X, (0, 0, 0, self.max_seq_length - N))
        Nm = self.max_seq_length

        # Token-level validity mask: (B, Nm), True for real tokens.
        if seq_lengths is None:
            # No seq_lengths supplied; assume every position up to the
            # original N is valid. (If N == Nm, no positions are padded.)
            if N == Nm:
                token_valid = None  # fast path: nothing is padded
            else:
                token_valid = torch.arange(Nm, device=X.device).unsqueeze(0).lt(N)
                token_valid = token_valid.expand(B, -1)
        else:
            if seq_lengths.shape != (B,):
                raise ValueError(
                    f"seq_lengths shape {tuple(seq_lengths.shape)} != ({B},)"
                )
            token_valid = (
                torch.arange(Nm, device=X.device).unsqueeze(0)
                < seq_lengths.unsqueeze(1)
            )  # (B, Nm)

        # Center / standardize using only real tokens for the EMA update.
        if self.center:
            if self.training:
                with torch.no_grad():
                    if token_valid is None:
                        flat = X.reshape(-1, D)
                    else:
                        flat = X[token_valid]
                    if flat.shape[0] > 0:
                        batch_mean = flat.mean(dim=0)
                        batch_var = flat.var(dim=0, unbiased=False)
                        if not bool(self.stats_initialized):
                            self.running_mean.copy_(batch_mean)
                            self.running_var.copy_(batch_var)
                            self.stats_initialized.fill_(True)
                        else:
                            m = self.center_momentum
                            self.running_mean.mul_(m).add_(batch_mean, alpha=1 - m)
                            self.running_var.mul_(m).add_(batch_var, alpha=1 - m)
            X = X - self.running_mean
            if self.standardize:
                X = X / (self.running_var + self.eps).sqrt()

        # Zero out pad positions AFTER centering so their contribution to
        # the chunk dot product is exactly zero. (Otherwise centering would
        # have shifted them to -running_mean/std.)
        if token_valid is not None:
            X = X * token_valid.unsqueeze(-1).to(X.dtype)

        # Reshape into chunks: (B, n_chunks, chunk_length, D) -> (B, n_chunks, L*D).
        chunks_flat = X.view(B, self.n_chunks, self.chunk_length, D).reshape(
            B, self.n_chunks, self.chunk_dim
        )

        # Chunk-level pad mask: True iff every token in the chunk is pad,
        # i.e. chunk start index >= seq_lengths[b].
        if token_valid is None:
            pad_mask = torch.zeros(
                B, self.n_chunks, dtype=torch.bool, device=X.device
            )
        else:
            # token_valid has shape (B, Nm). A chunk c is fully padded iff
            # token_valid is False for every token in [c*L, (c+1)*L).
            tv_chunks = token_valid.view(B, self.n_chunks, self.chunk_length)
            pad_mask = ~tv_chunks.any(dim=-1)  # (B, n_chunks)

        if single:
            chunks_flat = chunks_flat.squeeze(0)
            pad_mask = pad_mask.squeeze(0)
        return chunks_flat, pad_mask