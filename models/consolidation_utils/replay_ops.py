"""Backend-agnostic helpers for slot-prioritized replay.

The per-layer (ConsolidationModule + HippocampalBank) and the cross-layer
batched (StackedConsolidation) implementations store their bank state
differently, but the replay math is identical. These pure functions are
the single source of truth so the two paths cannot drift apart.
"""

import math

import torch


def summary_dim(num_heads: int, d_head: int, mode: str = "mean") -> int:
    """Decoder-input width for a given slot_value_summary mode."""
    base = 3 * num_heads * d_head
    return 2 * base if mode == "cls+mean" else base


def slot_value_summary(
    values_q: torch.Tensor,
    values_k: torch.Tensor,
    values_v: torch.Tensor,
    num_heads: int,
    d_head: int,
    n_chunks: int,
    chunk_length: int,
    mode: str = "mean",
) -> torch.Tensor:
    """Per-slot decoder-input summary, a deterministic selection/aggregation
    of already-stored bank values (no new storage).

    Each value tensor is (J, H, n_chunks, chunk_length*d_head). The last
    axis is token-major (token t occupies slice [t*d_head:(t+1)*d_head] of
    its chunk; see ChunkedRawKeyEncoder), so the CLS token (sequence
    position 0) is chunk 0, token 0.

    mode:
      'mean'     per-head mean over all chunks and tokens -> (J, 3*H*dh).
                 Global but dilutes the CLS token to ~1/N.
      'cls'      the CLS token's per-head value (chunk 0, token 0), the
                 slice most predictive of the CLS feature the anchor
                 targets -> (J, 3*H*dh). Pure selection of stored content.
      'cls+mean' concat of both -> (J, 6*H*dh).
    """
    J = values_q.shape[0]
    H, dh, C, L = num_heads, d_head, n_chunks, chunk_length
    outs = []
    for buf in (values_q, values_k, values_v):
        v = buf.view(J, H, C, L, dh)
        if mode == "mean":
            outs.append(v.mean(dim=(2, 3)).reshape(J, H * dh))
        elif mode == "cls":
            outs.append(v[:, :, 0, 0, :].reshape(J, H * dh))
        elif mode == "cls+mean":
            cls = v[:, :, 0, 0, :].reshape(J, H * dh)
            mean = v.mean(dim=(2, 3)).reshape(J, H * dh)
            outs.append(torch.cat([cls, mean], dim=-1))
        else:
            raise ValueError(
                f"unknown decoder summary mode {mode!r} "
                "(expected 'mean' / 'cls' / 'cls+mean')")
    return torch.cat(outs, dim=-1)


def histogram_disc(
    slot_class_hist: torch.Tensor,
    n_classes: int,
    eps: float,
) -> torch.Tensor:
    """Per-slot class-histogram peakiness in [0, 1]: 1 - H(Ĥ)/log(C).

    Even class spread -> ~0; one dominant class -> ~1; no hits -> 0.
    """
    total = slot_class_hist.sum(dim=-1)                        # (J,)
    p = slot_class_hist / total.clamp_min(eps).unsqueeze(-1)   # (J, C)
    ent = -(p * p.clamp_min(eps).log()).sum(dim=-1)            # (J,)
    log_c = math.log(n_classes) if n_classes > 1 else 1.0
    disc = 1.0 - ent / log_c
    return torch.where(total > eps, disc, torch.zeros_like(disc))


def priority_score(
    T: torch.Tensor,
    P: torch.Tensor,
    disc: torch.Tensor,
    gate: torch.Tensor,
    hit_count: torch.Tensor,
    w_single: float,
    w_cross: float,
    min_usage: float,
) -> torch.Tensor:
    """priority(j) = (w_single*sqrt(T) + w_cross*log1p(P)) * disc, gated and
    floored by lifetime usage. >= 0."""
    phi = T.clamp_min(0.0).sqrt()
    psi = torch.log1p(P.clamp_min(0.0))
    pr = (w_single * phi + w_cross * psi) * disc
    pr = pr * gate.to(pr.dtype)
    pr = torch.where(hit_count >= min_usage, pr, torch.zeros_like(pr))
    return pr.clamp_min(0.0)


def head_replay_step(
    *,
    decoder,
    decoder_optimizer,
    summaries: torch.Tensor,
    slot_class_hist: torch.Tensor,
    priority: torch.Tensor,
    head,
    loss_fn,
    m: int,
    replay_weight: float,
    anchor_weight: float,
    enable_ce: bool,
    anchor_j_star,
    anchor_feats,
    head_optimizer,
    eps: float,
    want_diagnostics: bool,
) -> dict:
    """Shared head-mode replay: priority-sampled CE on synthetic features
    + decoder anchoring MSE, backprop into {decoder, head}.

    summaries: (J, val_dim) per-slot decoder inputs (detached bank content).
    priority: (J,) >= 0 sampling weights. Returns diagnostics ({} if nothing
    ran).
    """
    loss = None
    diag: dict = {}

    if enable_ce and float(priority.sum()) > 0.0:
        draws = torch.multinomial(priority, m, replacement=True)      # (m,)
        feats = decoder(summaries[draws], add_noise=True)             # (m, D)
        hist = slot_class_hist[draws].clamp_min(eps)
        lbl = torch.multinomial(hist, 1).squeeze(1)                   # (m,)
        ce = loss_fn(head(feats), lbl)
        loss = replay_weight * ce
        if want_diagnostics:
            diag["replay_ce"] = float(ce.item())
            diag["replay_unique_slots"] = int(draws.unique().numel())
            diag["replay_mean_priority"] = float(priority[priority > 0].mean().item())

    if (anchor_weight > 0.0 and anchor_feats is not None
            and anchor_j_star is not None):
        pred = decoder(summaries[anchor_j_star.to(torch.long)], add_noise=False)
        a = torch.nn.functional.mse_loss(pred, anchor_feats.detach())
        loss = a * anchor_weight if loss is None else loss + anchor_weight * a
        if want_diagnostics:
            diag["replay_anchor_mse"] = float(a.item())

    if loss is None:
        return {}

    decoder_optimizer.zero_grad(set_to_none=True)
    if head_optimizer is not None:
        head_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    decoder_optimizer.step()
    if head_optimizer is not None:
        head_optimizer.step()
    return diag
