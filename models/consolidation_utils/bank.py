"""
HippocampalBank: stores keys and values for hippocampal-indexed consolidation.

Keys: (J, n_chunks, key_chunk_dim) — chunked input representation per slot.
Values: (J, H, n_chunks, value_chunk_dim) per Q/K/V — chunked per-head output
representation per slot.

Retrieval: position-aligned per-chunk cosine on keys, mean over chunks ->
scalar per (sample, slot). pad_mode applied at the input side to padded
chunks (-1 / 0 / compute) before the mean.

EMA update: per (slot, chunk) for keys, per (slot, head, chunk) for values,
both gated by the input pad mask so padded chunks never average in zeros.
The EMA step size scales with the total retrieval weight a (slot, chunk)
received this batch, capped at 1: a slot retrieved at softmax weight w
moves by (1-beta)*w, not the full (1-beta). With top_k=1 (w=1) this is
identical to plain EMA.

All state is buffers; no trainable parameters.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_ops import histogram_disc


class HippocampalBank(nn.Module):
    def __init__(
        self,
        n_chunks: int,
        key_chunk_dim: int,
        value_chunk_dim: int,
        num_heads: int,
        n_slots: int = 64,
        beta_key: float = 0.99,
        beta_value: float = 0.999,
        pad_mode: str = "neg_one",
        top_k: int = 1,
        retrieval_temperature: float = 0.1,
        blend_floor: float = 0.0,
        data_init: bool = False,
        dead_restart_interval: int = 0,
        dead_min_hits: float = 1.0,
        dead_protect_hits: float = 50.0,
        novelty_threshold: float = -1.0,
        n_classes: int = 0,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_chunks: C, chunks per sample.
            key_chunk_dim: chunk_length * D, flat input-chunk size.
            value_chunk_dim: chunk_length * d_head, flat per-head output-chunk size.
            num_heads: H.
            n_slots: J.
            beta_key, beta_value: EMA decays.
            pad_mode: 'neg_one' (default) | 'zero' | 'compute'. Applies to
                padded chunks in retrieve(). Output cosines exclude padded
                chunks regardless of this setting (see ConsolidationModule).
            top_k: number of slots to retrieve per sample. 1 = pure argmax
                (preserves prior behavior, no softmax overhead). >1 enables
                soft-attention retrieval: softmax-weighted blend over the
                top-k slots' values, weighted EMA update across the same
                top-k slots. Bounded by n_slots.
            retrieval_temperature: softmax temperature applied to per-slot
                mean similarities before top-k weighting. Lower = sharper
                (approaches argmax even at k>1). Ignored when top_k=1.
            blend_floor: lower bound on the weight-scaled EMA blend factor
                for touched (slot, chunk) pairs: m_eff = clamp(total_w,
                blend_floor, 1). 0 (default) keeps the corrected weight-
                scaled behavior; 1.0 reproduces the pre-2026-06-10
                semantics where any touch pulls at full (1-beta) strength
                (aggressive slot recruitment, but soft weights are ignored
                by the bank update).
            data_init: if True, slot keys are (re)seeded from the first
                batches of real encoded chunks (+1% noise) instead of
                keeping the random-normal init. Random keys are nearly
                orthogonal to data keys, so without this only slots that
                win the first noisy retrievals ever become reachable.
            dead_restart_interval: if > 0, every N update() calls slots
                whose hit_count grew by less than dead_min_hits since the
                last check are reseeded from random current-batch chunks
                (VQ-VAE-style dead-code restart). Their values reset to
                zero. 0 disables.
            dead_min_hits: minimum accumulated retrieval weight per
                interval for a slot to count as alive.
            dead_protect_hits: slots whose LIFETIME hit_count exceeds this
                are never reseeded, regardless of recent usage. Without
                this, slots holding earlier tasks' memories (heavily used
                then, unused now) would be flagged dead during later tasks
                and erased — defeating the cross-task purpose of the bank.
                Set to float('inf') to disable protection.
            novelty_threshold: if > -1, samples whose top-1 similarity
                falls below this threshold claim the least-used slots:
                the slot's key is overwritten with the sample's chunks
                and its values reset. -1 (default) disables. NOTE:
                enabling this adds one host-device sync per step.
            n_classes: total number of classes in the run. When > 0,
                enables slot-prioritized-replay bookkeeping: a per-slot
                class histogram (slot_class_hist) and a per-task hit
                counter (task_hit_count), both fed from the top-1 slot
                assignment + cached labels in update(). 0 (default)
                allocates nothing and keeps update() bit-identical to the
                pre-replay code path (these buffers are read-only to the
                replay scheduler — replay never calls update()).
            eps: cosine denominator floor.
            seed: RNG seed for key init.
        """
        super().__init__()
        for name, val in [
            ("n_chunks", n_chunks),
            ("key_chunk_dim", key_chunk_dim),
            ("value_chunk_dim", value_chunk_dim),
            ("num_heads", num_heads),
            ("n_slots", n_slots),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        if not 0.0 < beta_key < 1.0:
            raise ValueError(f"beta_key must be in (0,1), got {beta_key}")
        if not 0.0 < beta_value < 1.0:
            raise ValueError(f"beta_value must be in (0,1), got {beta_value}")
        if pad_mode not in ("neg_one", "zero", "compute"):
            raise ValueError(
                f"pad_mode must be 'neg_one'/'zero'/'compute', got {pad_mode!r}"
            )
        if top_k < 1 or top_k > n_slots:
            raise ValueError(
                f"top_k must be in [1, n_slots={n_slots}], got {top_k}"
            )
        if top_k > 1 and retrieval_temperature <= 0:
            raise ValueError(
                f"retrieval_temperature must be > 0, got {retrieval_temperature}"
            )
        if not 0.0 <= blend_floor <= 1.0:
            raise ValueError(f"blend_floor must be in [0,1], got {blend_floor}")
        if dead_restart_interval < 0:
            raise ValueError(
                f"dead_restart_interval must be >= 0, got {dead_restart_interval}"
            )
        if n_classes < 0:
            raise ValueError(f"n_classes must be >= 0, got {n_classes}")

        self.n_chunks = n_chunks
        self.key_chunk_dim = key_chunk_dim
        self.value_chunk_dim = value_chunk_dim
        self.num_heads = num_heads
        self.J = n_slots
        self.beta_key = beta_key
        self.beta_value = beta_value
        self.pad_mode = pad_mode
        self.top_k = top_k
        self.retrieval_temperature = float(retrieval_temperature)
        self.blend_floor = float(blend_floor)
        self.data_init = bool(data_init)
        self.dead_restart_interval = int(dead_restart_interval)
        self.dead_min_hits = float(dead_min_hits)
        self.dead_protect_hits = float(dead_protect_hits)
        self.novelty_threshold = float(novelty_threshold)
        self.n_classes = int(n_classes)
        self.eps = eps

        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            keys = torch.randn(n_slots, n_chunks, key_chunk_dim, generator=gen)
        else:
            keys = torch.randn(n_slots, n_chunks, key_chunk_dim)

        self.register_buffer("keys", keys)
        # Values initialized to zeros; bank fills via EMA during warmup.
        # Zero target -> ~zero cosine -> ~zero gradient until populated.
        for name in ("values_q", "values_k", "values_v"):
            self.register_buffer(
                name, torch.zeros(n_slots, num_heads, n_chunks, value_chunk_dim)
            )
        # Float so fractional top-k weights accumulate exactly. (A long
        # buffer truncated per-batch sums < 1 to zero, underreporting
        # usage precisely in the soft-retrieval regime.)
        self.register_buffer("hit_count", torch.zeros(n_slots))

        # Slot-prioritized-replay bookkeeping (allocated only when
        # n_classes > 0; otherwise the bank is byte-for-byte the pre-replay
        # module — no extra state_dict keys, no extra work in update()).
        #   slot_class_hist: (J, n_classes) top-1 class counts per slot,
        #       the source for the replay priority's discriminativeness
        #       term and for sampling a label per replayed slot.
        #   task_hit_count:  (J,) hits accrued during the CURRENT task,
        #       reset each task via reset_task_hits(). The cross-task term
        #       P_j = hit_count - task_hit_count is derived from it.
        if self.n_classes > 0:
            self.register_buffer(
                "slot_class_hist", torch.zeros(n_slots, self.n_classes)
            )
            self.register_buffer("task_hit_count", torch.zeros(n_slots))

        # Slot-recruitment state. keys_initialized is all-True when
        # data_init is off (the random init counts as initialized).
        self.register_buffer(
            "keys_initialized",
            torch.full((n_slots,), not data_init, dtype=torch.bool),
        )
        self.register_buffer("hit_snapshot", torch.zeros(n_slots))
        self.register_buffer(
            "updates_since_restart", torch.zeros((), dtype=torch.long)
        )
        # Python-side mirrors so the per-step checks never force a GPU
        # sync (same pattern as ConsolidationModule._step_py).
        self._all_seeded: bool = not data_init
        self._updates_py: int = 0
        self.register_load_state_dict_post_hook(self._sync_recruitment_state)

    @staticmethod
    def _sync_recruitment_state(module, incompatible_keys):
        module._all_seeded = bool(module.keys_initialized.all().item())
        module._updates_py = int(module.updates_since_restart.item())

    @torch.no_grad()
    def _seed_slots(
        self,
        slot_idx: torch.Tensor,
        donor_chunks: torch.Tensor,
        reset_values: bool = True,
    ) -> None:
        """Overwrite keys[slot_idx] with donor chunks (+1% noise)."""
        if slot_idx.numel() == 0:
            return
        noise = 0.01 * donor_chunks.std().clamp_min(self.eps)
        self.keys[slot_idx] = (
            donor_chunks + noise * torch.randn_like(donor_chunks)
        )
        if reset_values:
            for buf in (self.values_q, self.values_k, self.values_v):
                buf[slot_idx] = 0.0
        # A recycled slot must not replay its previous occupant's label:
        # clear its class histogram and per-task usage so it is not
        # eligible for replay until it earns fresh hits (disc -> 0).
        if self.n_classes > 0:
            self.slot_class_hist[slot_idx] = 0.0
            self.task_hit_count[slot_idx] = 0.0
        self.keys_initialized[slot_idx] = True

    @torch.no_grad()
    def retrieve(
        self,
        chunks: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Args:
            chunks: (B, C, key_chunk_dim) or (C, key_chunk_dim).
            pad_mask: (B, C) or (C,) bool, True where chunk is fully padded.
                None => no padding.

        Returns:
            s: (B,) — gate scalar. For top_k=1 this is the argmax slot's
                mean cosine. For top_k>1 it's the sum of softmax-weighted
                top-k mean cosines (=expected similarity under the
                retrieval distribution).
            j_top: (B, top_k) — slot indices in descending similarity.
            w_top: (B, top_k) — softmax weights over the top-k (rows
                sum to 1). With top_k=1 this is all ones.
            top_sims: (B, top_k) — raw per-slot mean cosines for the
                chosen slots (pre-softmax). Useful as a diagnostic to
                see whether the softmax is actually spreading mass or
                collapsing — comparing top_sims to w_top tells you that.
            S_q, S_k, S_v: (B, H, C, value_chunk_dim) — weighted blends
                of the top-k slots' values.
        """
        single = chunks.dim() == 2
        if single:
            chunks = chunks.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)
        if chunks.dim() != 3:
            raise ValueError(f"Expected chunks rank 2 or 3, got {chunks.dim()}")
        B, C, Dk = chunks.shape
        if (C, Dk) != (self.n_chunks, self.key_chunk_dim):
            raise ValueError(
                f"chunks shape (..., {C}, {Dk}) != expected "
                f"(..., {self.n_chunks}, {self.key_chunk_dim})"
            )
        if pad_mask is None:
            pad_mask = torch.zeros(B, C, dtype=torch.bool, device=chunks.device)
        elif pad_mask.shape != (B, C):
            raise ValueError(
                f"pad_mask shape {tuple(pad_mask.shape)} != ({B}, {C})"
            )

        s_n = F.normalize(chunks, dim=-1, eps=self.eps)        # (B, C, Dk)
        b_n = F.normalize(self.keys, dim=-1, eps=self.eps)     # (J, C, Dk)
        sims = torch.einsum("bcd,jcd->bjc", s_n, b_n)          # (B, J, C)

        if self.pad_mode != "compute":
            mask_exp = pad_mask.unsqueeze(1).expand(-1, self.J, -1)
            fill = -1.0 if self.pad_mode == "neg_one" else 0.0
            sims = torch.where(mask_exp, torch.full_like(sims, fill), sims)

        sim_per_slot = sims.mean(dim=-1)                       # (B, J)

        if self.top_k == 1:
            # Fast path: identical to prior argmax behavior, no softmax.
            s_top1, j_star = sim_per_slot.max(dim=-1)
            j_top = j_star.unsqueeze(-1)                       # (B, 1)
            top_sims = s_top1.unsqueeze(-1)                    # (B, 1)
            w_top = torch.ones_like(j_top, dtype=sim_per_slot.dtype)
            s = s_top1
            S_q = self.values_q[j_star]
            S_k = self.values_k[j_star]
            S_v = self.values_v[j_star]
        else:
            # Top-k soft-attention retrieval.
            k = self.top_k
            top_sims, j_top = sim_per_slot.topk(k, dim=-1)     # (B, k) each
            w_top = F.softmax(top_sims / self.retrieval_temperature, dim=-1)
            # Gate scalar = expected similarity under the softmax weighting.
            # Sum of w*sim is what the gradient weights into delta_W.
            s = (w_top * top_sims).sum(dim=-1)                 # (B,)

            # Gather per-slot values for the top-k, then weighted-sum.
            # values_q: (J, H, C, Dv). Index along J with j_top -> (B, k, H, C, Dv).
            H, Dv = self.num_heads, self.value_chunk_dim
            idx = j_top.view(B, k, 1, 1, 1).expand(B, k, H, C, Dv)
            S_q = (self.values_q.unsqueeze(0).expand(B, -1, -1, -1, -1)
                   .gather(1, idx) * w_top.view(B, k, 1, 1, 1)).sum(dim=1)
            S_k = (self.values_k.unsqueeze(0).expand(B, -1, -1, -1, -1)
                   .gather(1, idx) * w_top.view(B, k, 1, 1, 1)).sum(dim=1)
            S_v = (self.values_v.unsqueeze(0).expand(B, -1, -1, -1, -1)
                   .gather(1, idx) * w_top.view(B, k, 1, 1, 1)).sum(dim=1)

        if single:
            return (
                s.squeeze(0), j_top.squeeze(0), w_top.squeeze(0),
                top_sims.squeeze(0),
                S_q.squeeze(0), S_k.squeeze(0), S_v.squeeze(0),
            )
        return s, j_top, w_top, top_sims, S_q, S_k, S_v

    @torch.no_grad()
    def update(
        self,
        j_top: torch.Tensor,
        w_top: torch.Tensor,
        chunks: torch.Tensor,
        S_q: torch.Tensor,
        S_k: torch.Tensor,
        S_v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        top_sims: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """
        EMA-update top-k matched slots with softmax weighting. Inputs detached.

        For top_k=1 this is identical to the prior argmax-EMA update
        (w_top is all ones). For top_k>1 the same sample's chunks/values
        contribute fractionally to all k chosen slots, scaled by w_top.

        Keys: per (slot, chunk) weighted-mean over batch elements that
        hit that slot and have a non-padded chunk. Weighting is by
        w_top[b, slot_rank]. EMA blend; (slot, chunk) pairs that received
        zero total weight are untouched.

        The EMA blend factor scales with the total weight a (slot, chunk)
        received: new = old + (1-beta) * min(total_w, 1) * (mean - old).
        Without the min(total_w, 1) factor the weights cancel in the
        weighted mean whenever a single sample hits a slot, so a rank-3
        slot retrieved at w=0.05 would receive the same full-magnitude
        pull as the top-1 slot. With top_k=1 every contributing sample
        has w=1, total_w >= 1, and this reduces exactly to the plain
        EMA (full (1-beta) step toward the batch mean).

        Values: per (slot, head, chunk) weighted-mean over the same,
        with the same weight-scaled blend factor.

        Args:
            j_top: (B, k) slot indices.
            w_top: (B, k) per-slot weights (rows sum to 1 in soft mode,
                rows all-ones in top_k=1 mode).
            chunks: (B, C, key_chunk_dim).
            S_q, S_k, S_v: (B, H, C, value_chunk_dim).
            pad_mask: (B, C) bool. None => no padding.
            top_sims: (B, k) raw top-k similarities from retrieve();
                only consulted when novelty_threshold is enabled.
            labels: (B,) class labels for the batch. Only consulted when
                n_classes > 0, to accumulate the top-1 slot class
                histogram and per-task hit counter used by replay.
        """
        if j_top.dim() == 1:
            # Allow unbatched / legacy callers. Treat as (1, k).
            j_top = j_top.unsqueeze(0)
            w_top = w_top.unsqueeze(0)
            chunks = chunks.unsqueeze(0)
            S_q = S_q.unsqueeze(0)
            S_k = S_k.unsqueeze(0)
            S_v = S_v.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)
            if top_sims is not None:
                top_sims = top_sims.unsqueeze(0)
        if j_top.dim() != 2:
            raise ValueError(f"Expected j_top rank 1 or 2, got {j_top.dim()}")
        if w_top.shape != j_top.shape:
            raise ValueError(
                f"w_top shape {tuple(w_top.shape)} != j_top shape "
                f"{tuple(j_top.shape)}"
            )

        B, k = j_top.shape
        C, Dk, Dv = self.n_chunks, self.key_chunk_dim, self.value_chunk_dim
        H = self.num_heads
        if chunks.shape != (B, C, Dk):
            raise ValueError(
                f"chunks shape {tuple(chunks.shape)} != ({B}, {C}, {Dk})"
            )
        expected_val = (B, H, C, Dv)
        for name, S in [("S_q", S_q), ("S_k", S_k), ("S_v", S_v)]:
            if S.shape != expected_val:
                raise ValueError(
                    f"{name} shape {tuple(S.shape)} != {expected_val}"
                )
        if (j_top < 0).any() or (j_top >= self.J).any():
            raise ValueError(
                f"j_top out of range [0, {self.J}): "
                f"min={j_top.min().item()}, max={j_top.max().item()}"
            )
        if pad_mask is None:
            pad_mask = torch.zeros(B, C, dtype=torch.bool, device=chunks.device)
        elif pad_mask.shape != (B, C):
            raise ValueError(
                f"pad_mask shape {tuple(pad_mask.shape)} != ({B}, {C})"
            )

        valid = (~pad_mask).to(chunks.dtype)                   # (B, C)

        # ----- Data-dependent key init: seed any still-uninitialized
        # slots from this batch's encoded chunks. Random-normal keys are
        # nearly orthogonal to real data keys, so they are unreachable by
        # retrieval; seeding puts every slot in-distribution. Values are
        # already zero at construction, so no reset is needed here.
        if self.data_init and not self._all_seeded:
            uninit = (~self.keys_initialized).nonzero(as_tuple=True)[0]
            donors = torch.randperm(B, device=chunks.device)[:uninit.numel()]
            self._seed_slots(uninit[:donors.numel()], chunks[donors],
                             reset_values=False)
            self._all_seeded = bool(self.keys_initialized.all().item())

        # Flatten the (B, k) sample-rank dim into a single update list of
        # length B*k. Each "virtual sample" m=(b, r) contributes
        # w_top[b, r] * valid[b, c] to slot j_top[b, r].
        j_flat = j_top.reshape(B * k)                          # (B*k,)
        w_flat = w_top.reshape(B * k).to(chunks.dtype)         # (B*k,)
        # Broadcast B-major tensors across k.
        chunks_bk = chunks.unsqueeze(1).expand(B, k, C, Dk).reshape(B * k, C, Dk)
        S_q_bk = S_q.unsqueeze(1).expand(B, k, H, C, Dv).reshape(B * k, H, C, Dv)
        S_k_bk = S_k.unsqueeze(1).expand(B, k, H, C, Dv).reshape(B * k, H, C, Dv)
        S_v_bk = S_v.unsqueeze(1).expand(B, k, H, C, Dv).reshape(B * k, H, C, Dv)
        valid_bk = valid.unsqueeze(1).expand(B, k, C).reshape(B * k, C)
        # Effective per-(virtual sample, chunk) weight: w * valid.
        w_bkc = (w_flat.unsqueeze(-1) * valid_bk)              # (B*k, C)

        # ----- Per-(slot, chunk) total weight -----
        idx_count = j_flat.view(B * k, 1).expand(B * k, C)     # (B*k, C)
        count = torch.zeros(self.J, C, dtype=chunks.dtype, device=chunks.device)
        count.scatter_add_(0, idx_count, w_bkc)
        touched = count > self.eps                             # (J, C)
        denom = count.clamp_min(self.eps)
        # Weight-scaled blend factor: a slot that received total weight w<1
        # moves only w * (1-beta) of the way toward the batch mean. Capped
        # at 1 so multi-sample collisions keep the prior full-step behavior.
        # blend_floor raises the lower bound for touched pairs (1.0 =
        # pre-2026-06-10 full-strength recruitment).
        m_eff = torch.where(
            touched,
            count.clamp(min=self.blend_floor, max=1.0),
            torch.zeros_like(count),
        )                                                      # (J, C)

        # ----- Keys -----
        weighted_keys = chunks_bk * w_bkc.unsqueeze(-1)        # (B*k, C, Dk)
        idx_keys = j_flat.view(B * k, 1, 1).expand(B * k, C, Dk)
        sum_keys = torch.zeros_like(self.keys)
        sum_keys.scatter_add_(0, idx_keys, weighted_keys)
        mean_keys = sum_keys / denom.unsqueeze(-1)
        bk_beta = self.beta_key
        self.keys.add_(
            (1 - bk_beta) * m_eff.unsqueeze(-1) * (mean_keys - self.keys)
        )

        # ----- Values (Q, K, V) -----
        w_v = w_bkc.view(B * k, 1, C, 1)                       # (B*k, 1, C, 1)
        idx_v = j_flat.view(B * k, 1, 1, 1).expand(B * k, H, C, Dv)
        m_v = m_eff.view(self.J, 1, C, 1)
        denom_v = denom.view(self.J, 1, C, 1)
        bv = self.beta_value
        for buf, S in [
            (self.values_q, S_q_bk),
            (self.values_k, S_k_bk),
            (self.values_v, S_v_bk),
        ]:
            weighted = S * w_v
            sum_buf = torch.zeros_like(buf)
            sum_buf.scatter_add_(0, idx_v, weighted)
            mean_buf = sum_buf / denom_v
            buf.add_((1 - bv) * m_v * (mean_buf - buf))

        # ----- Hit count: float accumulation of retrieval weight. With
        # top_k=1 + w=1 this is exact (+1 per sample); with top_k>1 it
        # accumulates fractional involvement without truncation.
        hit_inc = torch.zeros(self.J, dtype=chunks.dtype, device=chunks.device)
        hit_inc.scatter_add_(0, j_flat, w_flat)
        self.hit_count.add_(hit_inc.to(self.hit_count.dtype))

        # ----- Replay bookkeeping (top-1 slot only). One unit-weight hit
        # per sample to its argmax slot, attributed to the sample's class.
        # Kept on the top-1 assignment (not the soft top-k spread) so the
        # class histogram tracks the slot's dominant occupant.
        if self.n_classes > 0 and labels is not None:
            j0 = j_top[:, 0]                                   # (B,) top-1 slot
            lbl = labels.reshape(-1).to(torch.long)
            if lbl.shape[0] == B:
                ones = torch.ones(B, device=chunks.device,
                                  dtype=self.slot_class_hist.dtype)
                flat_idx = j0 * self.n_classes + lbl
                self.slot_class_hist.view(-1).scatter_add_(0, flat_idx, ones)
                self.task_hit_count.scatter_add_(
                    0, j0, ones.to(self.task_hit_count.dtype))

        self.updates_since_restart += 1
        self._updates_py += 1

        # ----- Novelty-triggered claiming: samples whose best similarity
        # is below threshold overwrite the least-used slots with their own
        # keys. (Costs one host sync per step when enabled.)
        if self.novelty_threshold > -1.0 and top_sims is not None:
            novel = top_sims[:, 0] < self.novelty_threshold     # (B,)
            n_novel = int(novel.sum().item())
            if n_novel > 0:
                claim = torch.argsort(self.hit_count)[:n_novel]
                self._seed_slots(claim, chunks[novel])
                # Count the claim as a hit so consecutive batches don't
                # thrash the same slots.
                self.hit_count[claim] += 1.0

        # ----- Dead-slot restart: every interval, slots whose usage grew
        # by less than dead_min_hits are reseeded from random current-
        # batch chunks (VQ-VAE-style dead-code restart).
        if (self.dead_restart_interval > 0
                and self._updates_py >= self.dead_restart_interval):
            gained = self.hit_count - self.hit_snapshot
            dead = ((gained < self.dead_min_hits)
                    & (self.hit_count < self.dead_protect_hits)
                    ).nonzero(as_tuple=True)[0]
            if dead.numel() > 0:
                donors = torch.randint(
                    0, B, (dead.numel(),), device=chunks.device)
                self._seed_slots(dead, chunks[donors])
            self.hit_snapshot.copy_(self.hit_count)
            self.updates_since_restart.zero_()
            self._updates_py = 0

    @torch.no_grad()
    def slot_usage(self) -> torch.Tensor:
        return self.hit_count.clone()

    @torch.no_grad()
    def reset_task_hits(self) -> None:
        """Zero the per-task hit counter at the start of a task.

        No-op when replay bookkeeping is disabled (n_classes == 0). The
        lifetime hit_count and slot_class_hist are intentionally left
        intact: P_j = hit_count - task_hit_count then reflects all hits
        from prior tasks the moment the new task begins.
        """
        if self.n_classes > 0:
            self.task_hit_count.zero_()

    @torch.no_grad()
    def replay_stats(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Per-slot statistics for the replay priority score.

        Returns (T_j, P_j, disc_j, gate), each (J,):
            T_j   = current-task hits (task_hit_count).
            P_j   = prior-task hits (hit_count - task_hit_count, >= 0).
            disc_j= class-histogram peakiness in [0, 1]:
                    1 - entropy(Ĥ_j) / log(n_classes). A slot whose hits
                    span classes evenly -> ~0 (not worth replaying); a
                    slot dominated by one class -> ~1. Slots with no hits
                    get 0.
            gate  = bool, slot is initialized AND has been used at least
                    once. The min-usage floor and priority weighting are
                    applied by the replay scheduler, not here.
        """
        if self.n_classes <= 0:
            raise RuntimeError(
                "replay_stats() requires the bank to be built with "
                "n_classes > 0 (slot-prioritized replay enabled).")
        T = self.task_hit_count.clone()
        P = (self.hit_count - self.task_hit_count).clamp_min(0.0)
        disc = histogram_disc(self.slot_class_hist, self.n_classes, self.eps)
        gate = (self.hit_count > 0) & self.keys_initialized
        return T, P, disc, gate