"""
ConsolidationModule: per-layer hippocampal-indexed consolidation orchestrator.

Wraps Q/K/V weight tensors of one attention layer. Per training step:

  1. cache(X, Y_q, Y_k, Y_v, seq_lengths=None) during the layer's forward.
  2. After optimizer.step(), consolidate():
       a. Chunk-encode cached X -> (B, n_chunks, key_chunk_dim) + pad_mask.
       b. Bank retrieves best slot via per-chunk position-aligned cosine
          (mean over chunks; pad mode applied to padded chunks).
       c. Chunk-encode cached Y_q/Y_k/Y_v per-head ->
          (B, H, n_chunks, value_chunk_dim).
       d. Per-(b, h, c) cosine vs target; pad chunks zeroed out of the sum.
          Sum weighted by s_bank, autograd to G_Y, outer with X -> G_W.
       e. EMA-update bank keys per (slot, chunk) and values per (slot, head,
          chunk), masked by pad_mask.
       f. If past warmup, apply alpha * G_W to W.data.

Both key and value sides use ChunkedRawKeyEncoder (no SW compression).
Gradient flows through cosine directly — no SW backward.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .bank import HippocampalBank
from .decoder import SlotFeatureDecoder
from .encoders import ChunkedRawKeyEncoder
from .replay_ops import head_replay_step, priority_score, slot_value_summary, summary_dim


def separation_metrics(
    chunks: torch.Tensor,   # (B, n_chunks, key_chunk_dim)
    j_star: torch.Tensor,   # (B,)
    labels: torch.Tensor,   # (B,)
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Pairwise input-separation diagnostics.

    Two views of the same question ("do same-class samples cluster
    together in input space?"):

    (a) Continuous, encoder-only — cosine of encoded keys, averaged
        within-class and across-class. Tells us whether the key
        encoder separates classes BEFORE the bank quantizes:
          key_cos_within   = mean cos(key_i, key_j) over i!=j, c_i==c_j
          key_cos_across   = mean cos(key_i, key_j) over c_i!=c_j
          key_cos_margin   = within - across  (positive = separating)

    (b) Discrete, bank-mediated — slot agreement as a binary task:
          positive prediction = same slot, positive label = same class.
          TP/FP/TN/FN -> precision, recall, specificity, F1, accuracy,
          balanced_accuracy, MCC.

    Mean cosine across chunks (position-aligned) is the encoded key.
    """
    B, C, Dk = chunks.shape
    keys_per_chunk = torch.nn.functional.normalize(chunks, dim=-1, eps=eps)
    # Pairwise: sum_c <chunks[i,c], chunks[j,c]> / C
    pair_cos = torch.einsum("icd,jcd->ij", keys_per_chunk, keys_per_chunk) / C
    eye = torch.eye(B, dtype=torch.bool, device=pair_cos.device)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_class_off = same_class & ~eye
    diff_class = ~same_class

    out: Dict[str, float] = {}
    if same_class_off.any():
        out["key_cos_within"] = float(pair_cos[same_class_off].mean().item())
    else:
        out["key_cos_within"] = float("nan")
    if diff_class.any():
        out["key_cos_across"] = float(pair_cos[diff_class].mean().item())
    else:
        out["key_cos_across"] = float("nan")
    out["key_cos_margin"] = out["key_cos_within"] - out["key_cos_across"]

    # Slot agreement classification. Binary task per off-diagonal pair.
    same_slot = j_star.unsqueeze(0) == j_star.unsqueeze(1)
    off = ~eye
    pred_pos = same_slot[off]    # same slot
    true_pos = same_class[off]   # same class

    tp = int((pred_pos & true_pos).sum().item())
    fp = int((pred_pos & ~true_pos).sum().item())
    fn = int((~pred_pos & true_pos).sum().item())
    tn = int((~pred_pos & ~true_pos).sum().item())

    eps_f = 1e-12
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, eps_f)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    balanced_acc = 0.5 * (recall + specificity)
    # Matthews correlation coefficient — robust to class imbalance,
    # which we have (same-class pairs are far fewer than diff-class).
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / max(mcc_den, eps_f)

    out["sep_tp"] = tp
    out["sep_fp"] = fp
    out["sep_fn"] = fn
    out["sep_tn"] = tn
    out["sep_precision"] = float(precision)
    out["sep_recall"] = float(recall)
    out["sep_specificity"] = float(specificity)
    out["sep_f1"] = float(f1)
    out["sep_accuracy"] = float(accuracy)
    out["sep_balanced_acc"] = float(balanced_acc)
    out["sep_mcc"] = float(mcc)
    return out


class ConsolidationModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        d_head: int,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
        weight_v: torch.Tensor,
        max_seq_length: int,
        chunk_length: Optional[int] = None,
        pad_mode: str = "neg_one",
        n_slots: int = 64,
        beta_key: float = 0.99,
        beta_value: float = 0.999,
        alpha: float = 1e-3,
        warmup_steps: int = 16000,
        cache_projections: bool = True,
        center_keys: bool = True,
        standardize_keys: bool = True,
        center_values: bool = False,
        standardize_values: bool = False,
        center_momentum: float = 0.99,
        normalize_delta: bool = False,
        projection_dim: Optional[int] = None,
        projection_hidden: int = 128,
        projection_train_steps: int = 0,
        projection_train_always: bool = False,
        supcon_temperature: float = 0.1,
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
            input_dim: D, attention input feature dim.
            num_heads: H. d_head: per-head dim. W shape: (H*d_head, D).
            weight_q/k/v: external parameter views; updates via .data.add_().
            max_seq_length: N_max, pad target for both key and value encoders.
            chunk_length: tokens per chunk. None -> max_seq_length (single chunk).
            pad_mode: 'neg_one'/'zero'/'compute', applied only on the key
                (input) side in bank.retrieve(). Output cosines always
                exclude padded chunks.
            n_slots: J.
            beta_key, beta_value, alpha, warmup_steps: as before.
            center_keys, standardize_keys: per-feature stats for input encoder.
            center_values, standardize_values: per-feature stats for value
                encoders. Default False. Input centering escapes "average
                direction collapse" during retrieval; output side has no
                retrieval. Running stats also drift as W trains, creating
                a moving target inside the EMA target. Off by default.
            center_momentum, eps, seed: usual.
        """
        super().__init__()
        if input_dim < 1 or num_heads < 1 or d_head < 1:
            raise ValueError("input_dim/num_heads/d_head must be >= 1")
        if alpha < 0 or warmup_steps < 0:
            raise ValueError("alpha/warmup_steps must be non-negative")
        if chunk_length is None:
            chunk_length = max_seq_length
        if max_seq_length % chunk_length != 0:
            raise ValueError(
                f"max_seq_length ({max_seq_length}) must be divisible by "
                f"chunk_length ({chunk_length})"
            )

        expected_w_shape = (num_heads * d_head, input_dim)
        for name, W in [("weight_q", weight_q), ("weight_k", weight_k),
                        ("weight_v", weight_v)]:
            if tuple(W.shape) != expected_w_shape:
                raise ValueError(
                    f"{name} shape {tuple(W.shape)} != expected {expected_w_shape}"
                )

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_head = d_head
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.cache_projections = cache_projections
        self.normalize_delta = normalize_delta
        self.eps = eps
        self.max_seq_length = max_seq_length
        self.chunk_length = chunk_length

        self._external_weights = [weight_q, weight_k, weight_v]

        # Optional learned projection of X before chunking. When set, the
        # bank's key cosine acts on a (B, N, projection_dim)-shape feature
        # space instead of raw X. Trained by SupCon on cached labels.
        # The projection's input is detached from the main model — its loss
        # shapes only the projection, not the ViT trunk.
        self.projection_dim = projection_dim
        self.projection_train_steps = projection_train_steps
        self.projection_train_always = projection_train_always
        self.supcon_temperature = float(supcon_temperature)
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, projection_hidden),
                nn.GELU(),
                nn.Linear(projection_hidden, projection_dim),
            )
            encoder_input_dim = projection_dim
        else:
            self.projection = None
            encoder_input_dim = input_dim

        # Key (input) encoder: chunks (B, N, encoder_input_dim)
        # -> (B, n_chunks, L*encoder_input_dim).
        self.key_encoder = ChunkedRawKeyEncoder(
            input_dim=encoder_input_dim,
            max_seq_length=max_seq_length,
            chunk_length=chunk_length,
            center=center_keys,
            standardize=standardize_keys,
            center_momentum=center_momentum,
        )

        # Value (output) encoders: one per W. Each takes per-head reshape
        # (B*H, N, d_head) -> (B*H, n_chunks, L*d_head). Stats are per-d_head
        # feature, shared across heads, separate per Q/K/V.
        self.value_encoder_q = ChunkedRawKeyEncoder(
            input_dim=d_head,
            max_seq_length=max_seq_length,
            chunk_length=chunk_length,
            center=center_values,
            standardize=standardize_values,
            center_momentum=center_momentum,
        )
        self.value_encoder_k = ChunkedRawKeyEncoder(
            input_dim=d_head,
            max_seq_length=max_seq_length,
            chunk_length=chunk_length,
            center=center_values,
            standardize=standardize_values,
            center_momentum=center_momentum,
        )
        self.value_encoder_v = ChunkedRawKeyEncoder(
            input_dim=d_head,
            max_seq_length=max_seq_length,
            chunk_length=chunk_length,
            center=center_values,
            standardize=standardize_values,
            center_momentum=center_momentum,
        )

        n_chunks = self.key_encoder.n_chunks
        key_chunk_dim = self.key_encoder.chunk_dim
        value_chunk_dim = self.value_encoder_q.chunk_dim

        self.bank = HippocampalBank(
            n_chunks=n_chunks,
            key_chunk_dim=key_chunk_dim,
            value_chunk_dim=value_chunk_dim,
            num_heads=num_heads,
            n_slots=n_slots,
            beta_key=beta_key,
            beta_value=beta_value,
            pad_mode=pad_mode,
            top_k=top_k,
            retrieval_temperature=retrieval_temperature,
            blend_floor=blend_floor,
            data_init=data_init,
            dead_restart_interval=dead_restart_interval,
            dead_min_hits=dead_min_hits,
            dead_protect_hits=dead_protect_hits,
            novelty_threshold=novelty_threshold,
            n_classes=n_classes,
            eps=eps,
            seed=seed,
        )

        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))
        # Python-side mirror of step_count so per-step warmup / projection-
        # window checks never force a GPU sync via .item(). Kept consistent
        # with the buffer on every increment and on state_dict load.
        self._step_py: int = 0
        self.register_load_state_dict_post_hook(self._sync_step_py)

        # Value encoders with stats disabled (the default) are pure
        # pad+chunk reshapes, identical across Q/K/V, so the three cosine
        # graphs can be stacked into one autograd call.
        self._value_encoders_stateless = not (center_values or standardize_values)

        self._cached_X: Optional[torch.Tensor] = None
        self._cached_Y_q: Optional[torch.Tensor] = None
        self._cached_Y_k: Optional[torch.Tensor] = None
        self._cached_Y_v: Optional[torch.Tensor] = None
        self._cached_seq_lengths: Optional[torch.Tensor] = None
        self._cached_labels: Optional[torch.Tensor] = None

        # Slot-prioritized-replay decoder (head-mode). Created lazily via
        # setup_decoder(), and only on the chosen decode layer — most
        # modules never hold one. Its optimizer persists across tasks
        # (like the projection optimizer; the main optimizer is reset).
        self.decoder: Optional[SlotFeatureDecoder] = None
        self.decoder_optimizer: Optional[torch.optim.Optimizer] = None
        self.decoder_val_dim: Optional[int] = None
        self.decoder_summary_mode: str = "mean"
        # Top-1 decode-layer slot assignment from the most recent
        # consolidate() call, kept (cheaply, no host sync) so the wrapper
        # can align replay anchoring to the live batch.
        self._last_j_star: Optional[torch.Tensor] = None
        # Per-W consolidation step (post-normalize, pre-alpha) from the most
        # recent applied consolidate(); re-ascended by weight-mode replay at
        # a floored, schedule-independent alpha. None during warmup / when no
        # step was applied.
        self._last_step_qkv = None

    @staticmethod
    def _sync_step_py(module, incompatible_keys):
        module._step_py = int(module.step_count.item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cache(
        self,
        X: torch.Tensor,
        Y_q: Optional[torch.Tensor] = None,
        Y_k: Optional[torch.Tensor] = None,
        Y_v: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self._cached_X = X.detach()
        self._cached_seq_lengths = (
            seq_lengths.detach() if seq_lengths is not None else None
        )
        self._cached_labels = labels.detach() if labels is not None else None
        if self.cache_projections:
            if Y_q is None or Y_k is None or Y_v is None:
                raise ValueError(
                    "cache_projections=True requires Y_q, Y_k, Y_v."
                )
            self._cached_Y_q = Y_q.detach()
            self._cached_Y_k = Y_k.detach()
            self._cached_Y_v = Y_v.detach()

    @torch.no_grad()
    def on_task_start(self) -> None:
        """
        Reset per-task state. Currently a no-op — kept for interface
        stability across methods. lr scheduling state is owned by the
        caller (see consolidate's lr_anchor argument).
        """
        pass

    @torch.no_grad()
    def consolidate(
        self,
        lr_now: Optional[float] = None,
        lr_anchor: Optional[float] = None,
        compute_separation_metrics: bool = False,
        want_diagnostics: bool = True,
    ) -> Dict[str, object]:
        """
        Apply one consolidation step.

        Args:
            lr_now: current optimizer LR. If provided alongside lr_anchor,
                scales effective alpha by lr_now / lr_anchor.
            lr_anchor: reference LR for the scaling. Should be supplied by
                the caller and stay constant across a task (e.g. set to
                base_lr at the start of each task). If lr_now is given
                without lr_anchor, no scaling applies.
            compute_separation_metrics: if True and labels were cached,
                compute pairwise input-separation diagnostics (s_within
                / s_across at the key level, plus slot-agreement binary
                classification metrics keyed by class identity). Off
                by default since it adds a (B,B) chunk-cosine and full-
                bank similarity recompute per call. Enable on log steps
                only.
            want_diagnostics: if False, skip all host-side diagnostic
                materialization (.item()/.cpu() calls force a GPU sync;
                across 12 layers that is >100 syncs per training step)
                and return {}. The consolidation update itself is
                identical. Pass True on log steps only.
        """
        if self._cached_X is None:
            raise RuntimeError("No cached input. Call cache() before consolidate().")
        # Cleared each call; set only when a step is actually applied below,
        # so weight-mode replay never re-ascends a stale or warmup delta.
        self._last_step_qkv = None

        bank_dtype = self.bank.keys.dtype
        X = self._cached_X.to(dtype=bank_dtype)
        cY_q = self._cached_Y_q.to(dtype=bank_dtype) if self._cached_Y_q is not None else None
        cY_k = self._cached_Y_k.to(dtype=bank_dtype) if self._cached_Y_k is not None else None
        cY_v = self._cached_Y_v.to(dtype=bank_dtype) if self._cached_Y_v is not None else None
        seq_lengths = self._cached_seq_lengths
        labels = self._cached_labels

        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)
        if X.dim() != 3:
            raise RuntimeError(f"Cached X has rank {X.dim()}, expected 2 or 3.")
        B, N, D = X.shape
        H, dh = self.num_heads, self.d_head
        if D != self.input_dim:
            raise RuntimeError(f"X feature dim {D} != input_dim {self.input_dim}")

        # 1. Key encode. If a projection is configured, apply it first.
        # Detach the projection's output here so the bank retrieval path
        # never propagates gradients into the projection — those flow only
        # through compute_supcon_loss() if requested by the caller.
        if self.projection is not None:
            with torch.no_grad():
                X_for_keys = self.projection(X)
        else:
            X_for_keys = X
        chunks, pad_mask = self.key_encoder(X_for_keys, seq_lengths=seq_lengths)

        # 2. Bank retrieve. With top_k=1 this is the prior argmax behavior;
        # with top_k>1 it's softmax-weighted soft attention over top-k slots.
        s, j_top, w_top, top_sims, S_q_tgt, S_k_tgt, S_v_tgt = self.bank.retrieve(
            chunks, pad_mask
        )
        # s: (B,), j_top: (B, k), w_top: (B, k), top_sims: (B, k),
        # S_*_tgt: (B, H, n_chunks, value_chunk_dim) — already weighted-blended.
        # For backward-compat diagnostics, keep j_star as the top-1 slot.
        j_star = j_top[:, 0]
        # Stash for replay anchoring (cheap reference; no host sync).
        self._last_j_star = j_star.detach()

        # 3. Resolve Y (cached or recompute)
        Y_q = self._resolve_Y(cY_q, X, self._external_weights[0], "Y_q", B, N)
        Y_k = self._resolve_Y(cY_k, X, self._external_weights[1], "Y_k", B, N)
        Y_v = self._resolve_Y(cY_v, X, self._external_weights[2], "Y_v", B, N)
        # Each: (B, N, H, dh)

        # 4. Per-W: chunked output cosine vs target, autograd to W.
        # When the value encoders are stateless (default), the three
        # cosine graphs are stacked into a single autograd call.
        if self._value_encoders_stateless:
            (delta_W_q, delta_W_k, delta_W_v), (S_q_s, S_k_s, S_v_s), vcoses = \
                self._delta_and_sig_stacked(
                    Y_q, Y_k, Y_v, S_q_tgt, S_k_tgt, S_v_tgt, s, X,
                    pad_mask, seq_lengths, want_diagnostics,
                )
            vcos_q, vcos_k, vcos_v = vcoses
        else:
            delta_W_q, S_q_s, vcos_q = self._delta_and_sig(
                Y_q, S_q_tgt, s, X, pad_mask, seq_lengths,
                self.value_encoder_q, want_diagnostics,
            )
            delta_W_k, S_k_s, vcos_k = self._delta_and_sig(
                Y_k, S_k_tgt, s, X, pad_mask, seq_lengths,
                self.value_encoder_k, want_diagnostics,
            )
            delta_W_v, S_v_s, vcos_v = self._delta_and_sig(
                Y_v, S_v_tgt, s, X, pad_mask, seq_lengths,
                self.value_encoder_v, want_diagnostics,
            )

        # 5. EMA-update bank with the same top-k+weights used for retrieval.
        # labels (when present) feed the replay class histogram / per-task
        # hit counter; ignored by the bank when n_classes == 0.
        self.bank.update(
            j_top, w_top, chunks, S_q_s, S_k_s, S_v_s, pad_mask=pad_mask,
            top_sims=top_sims, labels=labels,
        )

        # 6. Apply weight update (gated by warmup). The warmup check uses
        # the python-side counter so no GPU sync happens off log steps.
        self.step_count += 1
        self._step_py += 1
        in_warmup = self._step_py <= self.warmup_steps
        # Pre-normalization magnitudes; needed on-device when
        # normalize_delta is on, materialized to floats only for logging.
        dq_norm_t = delta_W_q.norm()
        dk_norm_t = delta_W_k.norm()
        dv_norm_t = delta_W_v.norm()

        # Resolve effective alpha. If lr_now and lr_anchor are both given,
        # scale by lr_now / lr_anchor. Anchor is supplied by the caller
        # (e.g. base_lr captured at task start) so consolidation isn't
        # coupled to LR-warmup-ramp transients.
        alpha_eff = self.alpha
        lr_ratio = 1.0
        if not in_warmup and lr_now is not None and lr_anchor is not None:
            lr_ratio = float(lr_now) / max(float(lr_anchor), 1e-12)
            alpha_eff = self.alpha * lr_ratio

        if not in_warmup and alpha_eff > 0:
            if self.normalize_delta:
                # Per-W unit normalization: each weight tensor moves by
                # exactly alpha_eff (Frobenius). Decouples alpha from
                # per-layer gradient magnitude variation and from encoder choice.
                step_q = delta_W_q / (dq_norm_t + self.eps)
                step_k = delta_W_k / (dk_norm_t + self.eps)
                step_v = delta_W_v / (dv_norm_t + self.eps)
            else:
                step_q, step_k, step_v = delta_W_q, delta_W_k, delta_W_v
            self._external_weights[0].data.add_(alpha_eff * step_q)
            self._external_weights[1].data.add_(alpha_eff * step_k)
            self._external_weights[2].data.add_(alpha_eff * step_v)
            # Stash the (detached) normalized step so weight-mode replay can
            # re-ascend this same consolidation gradient at a floored alpha.
            self._last_step_qkv = (step_q, step_k, step_v)

        if not want_diagnostics:
            self._clear_cache()
            return {}

        dq_norm = float(dq_norm_t.item())
        dk_norm = float(dk_norm_t.item())
        dv_norm = float(dv_norm_t.item())

        # Optional pairwise separation diagnostics, computed only on log
        # steps. Uses labels cached alongside X (no-op if labels absent).
        sep_metrics = {}
        if compute_separation_metrics and labels is not None and chunks.shape[0] >= 2:
            sep_metrics = self._compute_separation_metrics(chunks, j_star, labels)

        # Top-k retrieval diagnostics. Per-rank means of both raw
        # similarities and softmax weights show whether the softmax is
        # truly distributed across the top-k or collapsing onto rank 0.
        # Entropy of w_top is the most compact summary: 0 = argmax-like,
        # log(k) = uniform.
        k = self.bank.top_k
        with torch.no_grad():
            top_sims_mean_per_rank = top_sims.mean(dim=0).cpu().tolist()
            w_top_mean_per_rank = w_top.mean(dim=0).cpu().tolist()
            if k > 1:
                w_safe = w_top.clamp_min(self.eps)
                ent = -(w_safe * w_safe.log()).sum(dim=-1)     # (B,)
                top_sim_entropy_mean = float(ent.mean().item())
                top_sim_entropy_max = float(ent.max().item())
                top_sim_entropy_min = float(ent.min().item())
                unique_topk_slots = int(j_top.unique().numel())
            else:
                top_sim_entropy_mean = 0.0
                top_sim_entropy_max = 0.0
                top_sim_entropy_min = 0.0
                unique_topk_slots = int(j_star.unique().numel())

        diagnostics: Dict[str, object] = {
            "step": self._step_py,
            "in_warmup": bool(in_warmup),
            "s_mean": float(s.mean().item()),
            "s_min": float(s.min().item()),
            "s_max": float(s.max().item()),
            "j_star": j_star.cpu(),
            # Pre-normalization gradient magnitudes (what the cosine-grad
            # path produced). Effective step magnitude is alpha_eff when
            # normalize_delta is on, else alpha_eff * delta_norm.
            "delta_norm_q": dq_norm,
            "delta_norm_k": dk_norm,
            "delta_norm_v": dv_norm,
            # Effective alpha after LR-ratio scaling and the ratio itself
            # (1.0 when lr_now is None or anchor isn't set yet).
            "alpha_eff": float(alpha_eff),
            "lr_ratio": float(lr_ratio),
            # Value-side cosine drives the actual gradient (not the input-
            # side s, which is just the per-sample gate). Mean over (B, H, C),
            # excluding pad chunks.
            "vcos_q": vcos_q,
            "vcos_k": vcos_k,
            "vcos_v": vcos_v,
            "unique_slots_hit": int(j_star.unique().numel()),
            # Top-k specific.
            "top_k": k,
            "unique_topk_slots": unique_topk_slots,
            "top_sims_mean_per_rank": top_sims_mean_per_rank,
            "w_top_mean_per_rank": w_top_mean_per_rank,
            "w_top_entropy_mean": top_sim_entropy_mean,
            "w_top_entropy_max": top_sim_entropy_max,
            "w_top_entropy_min": top_sim_entropy_min,
            **sep_metrics,
        }

        self._clear_cache()
        return diagnostics

    def projection_is_training_now(self) -> bool:
        """True iff projection exists AND is in its training window."""
        if self.projection is None:
            return False
        if self.projection_train_always:
            return True
        return self._step_py < self.projection_train_steps

    def compute_supcon_loss(self) -> Optional[torch.Tensor]:
        """
        Supervised contrastive loss on the projected key.

        Uses the cached X and labels from the last cache() call. Returns
        a scalar loss tensor with autograd through the projection, or
        None if no projection is configured, no labels were cached, the
        projection is past its training window, or the batch lacks at
        least one same-class positive pair.

        Gradient detaches X from the main model — only projection params
        receive supcon gradients.

        Definition: per anchor i, summing over all other samples j in the
        batch as candidates,
            L_i = -log( sum_{j: y_j=y_i, j!=i} exp(s_ij/tau)
                       / sum_{j: j!=i} exp(s_ij/tau) )
        averaged over anchors that have at least one positive.
        s_ij is cosine similarity between the projected, chunk-averaged
        key vectors of samples i and j.
        """
        if not self.projection_is_training_now():
            return None
        if self._cached_X is None or self._cached_labels is None:
            return None

        X = self._cached_X.detach().to(dtype=self.bank.keys.dtype)
        labels = self._cached_labels
        if X.dim() == 2:
            X = X.unsqueeze(0)
            labels = labels.unsqueeze(0)
        B = X.shape[0]
        if B < 2:
            return None

        # Project, chunk via the key encoder (in eval-stats mode to avoid
        # polluting the running mean/var with the same X twice per step),
        # then L2-normalize each chunk and average over chunks -> per-
        # sample key vector.
        Xp = self.projection(X)                          # (B, N, P)
        was_training = self.key_encoder.training
        self.key_encoder.eval()  # don't update running stats on this pass
        try:
            chunks, pad_mask = self.key_encoder(
                Xp, seq_lengths=self._cached_seq_lengths,
            )
        finally:
            if was_training:
                self.key_encoder.train()
        # chunks: (B, n_chunks, P*L); pad_mask: (B, n_chunks)
        chunks_n = torch.nn.functional.normalize(chunks, dim=-1, eps=self.eps)
        # Mask out fully-padded chunks before averaging.
        valid = (~pad_mask).to(chunks_n.dtype).unsqueeze(-1)  # (B, C, 1)
        denom = valid.sum(dim=1).clamp_min(1.0)               # (B, 1)
        key = (chunks_n * valid).sum(dim=1) / denom           # (B, P*L)
        key = torch.nn.functional.normalize(key, dim=-1, eps=self.eps)

        # Pairwise similarity / temperature.
        logits = (key @ key.t()) / self.supcon_temperature    # (B, B)
        # Mask out self.
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye, float("-inf"))

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        same_class = same_class & ~eye

        # Numerical stability: subtract row max.
        logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)                        # (B, B)
        # Per-row partition function (denominator).
        denom_p = exp_logits.sum(dim=-1).clamp_min(self.eps)  # (B,)

        # Per-row sum of exp over positive pairs.
        pos_exp = (exp_logits * same_class.to(exp_logits.dtype)).sum(dim=-1)
        # Only anchors with at least one positive contribute.
        has_pos = same_class.any(dim=-1)
        if not has_pos.any():
            return None
        per_row_loss = -torch.log(pos_exp.clamp_min(self.eps) / denom_p)
        loss = per_row_loss[has_pos].mean()
        return loss

    # ------------------------------------------------------------------
    # Slot-prioritized replay: decoder + anchoring (head-mode generator)
    # ------------------------------------------------------------------

    def setup_decoder(
        self,
        feature_dim: Optional[int] = None,
        hidden: int = 384,
        noise: float = 0.0,
        lr: float = 1e-3,
        summary_mode: str = "mean",
    ) -> SlotFeatureDecoder:
        """Build the head-mode decoder + its (persistent) optimizer.

        Called by the wrapper for the decode layer only. The decoder maps
        a per-slot value summary (width set by summary_mode; see
        slot_value_summary) to a synthetic CLS feature of width feature_dim
        (defaults to input_dim D).
        """
        if feature_dim is None:
            feature_dim = self.input_dim
        self.decoder_summary_mode = summary_mode
        self.decoder_val_dim = summary_dim(self.num_heads, self.d_head, summary_mode)
        self.decoder = SlotFeatureDecoder(
            self.decoder_val_dim, feature_dim, hidden=hidden, noise=noise,
        ).to(self.bank.keys.device)
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(), lr=lr, weight_decay=0.0)
        return self.decoder

    @torch.no_grad()
    def slot_value_matrix(self) -> torch.Tensor:
        """Per-slot value summary for decoder input: (J, 3*H*d_head).

        For each of the stored Q/K/V value tensors (J, H, n_chunks,
        L*d_head), reshape the last axis to (L, d_head) and mean over
        chunks and token positions, yielding a per-head (H, d_head)
        content vector; concatenate Q/K/V. This is a deterministic
        function of already-stored content (no new storage), and is the
        decoder's input. (Padded token positions are stored as zeros and
        dilute the mean slightly; nil for the default single-chunk,
        unpadded config.)
        """
        H, dh, C, L = self.num_heads, self.d_head, self.bank.n_chunks, self.chunk_length
        if self.bank.value_chunk_dim != L * dh:
            raise RuntimeError(
                f"value_chunk_dim {self.bank.value_chunk_dim} != L*d_head "
                f"{L * dh}; slot_value_matrix assumes L*d_head packing")
        return slot_value_summary(
            self.bank.values_q, self.bank.values_k, self.bank.values_v,
            H, dh, C, L, mode=self.decoder_summary_mode)

    @torch.no_grad()
    def reset_task_hits(self) -> None:
        """Zero the per-task hit counter (delegates to the bank)."""
        self.bank.reset_task_hits()

    def anchor_loss(
        self,
        j_star: torch.Tensor,
        real_feats: torch.Tensor,
        add_noise: bool = False,
    ) -> torch.Tensor:
        """MSE(D(value_{j_star}), real CLS feature), per sample.

        Ties the decoder to the real feature distribution: for each sample
        in the live batch we know its decode-layer slot j_star, its real
        pre-head CLS feature, and (implicitly) its label. The real feature
        is used transiently and never stored. Gradients flow into the
        decoder only (slot summaries are detached bank content).

        add_noise defaults False: the anchoring target is the real (mean)
        feature, so a deterministic decode is what should match it.
        """
        if self.decoder is None:
            raise RuntimeError("setup_decoder() must run before anchor_loss()")
        summaries = self.slot_value_matrix()                  # (J, val_dim)
        inp = summaries[j_star.to(torch.long)]                # (B, val_dim)
        pred = self.decoder(inp, add_noise=add_noise)         # (B, D)
        return torch.nn.functional.mse_loss(pred, real_feats.detach())

    @torch.no_grad()
    def replay_priority(
        self,
        w_single: float,
        w_cross: float,
        min_usage: float,
    ) -> torch.Tensor:
        """Per-slot replay priority (J,), >= 0.

        priority(j) = (w_single*sqrt(T_j) + w_cross*log1p(P_j)) * disc(j),
        zeroed for slots that are uninitialized/unused (gate) or below the
        lifetime-usage floor. sqrt on current-task hits, log1p on prior-
        task hits: prior-task slots are replayed but softly, so new slots
        still consolidate.
        """
        T, P, disc, gate = self.bank.replay_stats()
        return priority_score(T, P, disc, gate, self.bank.hit_count,
                              w_single, w_cross, min_usage)

    def replay_step(
        self,
        head,
        loss_fn,
        m: int,
        replay_weight: float,
        anchor_weight: float,
        w_single: float,
        w_cross: float,
        min_usage: float,
        enable_ce: bool,
        anchor_j_star: Optional[torch.Tensor] = None,
        anchor_feats: Optional[torch.Tensor] = None,
        head_optimizer=None,
        want_diagnostics: bool = False,
    ) -> Dict[str, object]:
        """One head-mode replay step on the decode layer.

        Builds a loss with two parts and backprops into {decoder, head}
        (the backbone is never in this graph, so it is frozen by
        construction):
          CE:     m prioritized slots -> decoder(+noise) -> head ->
                  CE vs a label sampled from each slot's class histogram.
          anchor: MSE(decoder(value_{j_star}), real CLS feat) on the live
                  batch, keeping the decoder tied to the real manifold.

        The CE term fires only when enable_ce (replay starts at task 2)
        and at least one slot is replay-eligible; anchoring runs whenever
        a live (j_star, feats) pair is supplied, so the decoder is trained
        from task 1 before replay begins.

        Returns diagnostics ({} off log steps or when nothing ran).
        """
        if self.decoder is None:
            raise RuntimeError("setup_decoder() must run before replay_step()")
        return head_replay_step(
            decoder=self.decoder, decoder_optimizer=self.decoder_optimizer,
            summaries=self.slot_value_matrix(),
            slot_class_hist=self.bank.slot_class_hist,
            priority=self.replay_priority(w_single, w_cross, min_usage),
            head=head, loss_fn=loss_fn, m=m,
            replay_weight=replay_weight, anchor_weight=anchor_weight,
            enable_ce=enable_ce,
            anchor_j_star=anchor_j_star, anchor_feats=anchor_feats,
            head_optimizer=head_optimizer, eps=self.eps,
            want_diagnostics=want_diagnostics)

    @torch.no_grad()
    def weight_replay(self, alpha_floor: float) -> bool:
        """Weight-mode replay: re-ascend the last consolidation gradient.

        Re-applies the most recent consolidate() step (the same per-W
        normalized delta) at a floored alpha, decoupled from the cosine LR
        schedule (the normal a_eff = alpha*lr_ratio anneals to ~1e-6
        mid-task; this keeps consolidation pulling at full strength to
        counter representation drift). Read-only to the bank: no retrieval,
        no EMA update, no hit_count change.

        Returns True if a step was applied (False during warmup or when no
        consolidate step was applied this batch).
        """
        if self._last_step_qkv is None or alpha_floor <= 0.0:
            return False
        step_q, step_k, step_v = self._last_step_qkv
        self._external_weights[0].data.add_(alpha_floor * step_q)
        self._external_weights[1].data.add_(alpha_floor * step_k)
        self._external_weights[2].data.add_(alpha_floor * step_v)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_Y(self, cached, X, W, name, B, N):
        H, dh = self.num_heads, self.d_head
        if self.cache_projections:
            if cached is None:
                raise RuntimeError(
                    f"cache_projections=True but {name} not provided in cache()."
                )
            Y = cached
            if Y.dim() == 2:
                Y = Y.unsqueeze(0)
            if tuple(Y.shape) != (B, N, H * dh):
                raise RuntimeError(
                    f"Cached {name} shape {tuple(Y.shape)} != ({B}, {N}, {H*dh})"
                )
        else:
            Y = X @ W.T
        return Y.view(B, N, H, dh)

    def _delta_and_sig(
        self,
        Y_per_head: torch.Tensor,    # (B, N, H, dh)
        S_target: torch.Tensor,      # (B, H, n_chunks, value_chunk_dim)
        s_bank: torch.Tensor,        # (B,)
        X: torch.Tensor,             # (B, N, D)
        pad_mask: torch.Tensor,      # (B, n_chunks)
        seq_lengths: Optional[torch.Tensor],
        value_encoder: ChunkedRawKeyEncoder,
        want_diagnostics: bool = True,
    ):
        """
        Compute per-W delta from chunked per-head cosine + autograd.

        Returns:
            G_W: (H*dh, D), pre-alpha weight delta.
            S_student: (B, H, n_chunks, value_chunk_dim), detached.
            mean_vcos: float, mean per-(b,h,c) cosine over non-padded chunks
                (0.0 when want_diagnostics is False — skips a GPU sync).
                This is the actual gradient driver; the input-side s in
                diagnostics is just the per-sample gate. Useful for
                noticing when targets have nothing in common with student
                (vcos near 0) or are aligned (vcos -> 1).
        """
        B, N, H, dh = Y_per_head.shape

        # Expand seq_lengths to (B*H,) for the per-head reshape, since each
        # head sees the same per-sample length.
        if seq_lengths is not None:
            sl_flat = seq_lengths.unsqueeze(-1).expand(B, H).reshape(B * H)
        else:
            sl_flat = None

        with torch.enable_grad():
            Y_leaf = Y_per_head.detach().requires_grad_(True)
            # (B, N, H, dh) -> (B, H, N, dh) -> (B*H, N, dh)
            Y_flat = Y_leaf.permute(0, 2, 1, 3).reshape(B * H, N, dh)
            # Chunk + (optional) center & standardize + zero pad positions.
            Y_chunks_flat, _ = value_encoder(Y_flat, seq_lengths=sl_flat)
            # (B*H, n_chunks, value_chunk_dim)
            nc = Y_chunks_flat.shape[1]
            Dv = Y_chunks_flat.shape[2]
            Y_chunks = Y_chunks_flat.reshape(B, H, nc, Dv)

            # Per-(b, h, c) cosine vs target.
            dot = (Y_chunks * S_target).sum(dim=-1)             # (B, H, nc)
            ny = Y_chunks.norm(dim=-1).clamp_min(self.eps)
            nt = S_target.norm(dim=-1).clamp_min(self.eps)
            cos_bhc = dot / (ny * nt)                           # (B, H, nc)

            # Exclude padded chunks from the gradient (their Y positions
            # are zeroed by the encoder; gradient there is uninformative).
            valid = (~pad_mask).to(cos_bhc.dtype).unsqueeze(1)  # (B, 1, nc)
            cos_bhc = cos_bhc * valid

            # Sum weighted by s_bank per batch element.
            total = (cos_bhc * s_bank.view(-1, 1, 1)).sum()
            (G_Y,) = torch.autograd.grad(total, Y_leaf, create_graph=False)
            # G_Y: (B, N, H, dh)

        # Mean cosine over valid (non-padded) (b, h, c) entries, for
        # diagnostics only.
        mean_vcos = 0.0
        if want_diagnostics:
            with torch.no_grad():
                n_valid = (valid.sum() * H).clamp_min(1.0)
                mean_vcos = float((cos_bhc.sum() / n_valid).item())

        # Outer product: G_W[h*dh+d, e] = sum_{b, n} G_Y[b,n,h,d] * X[b,n,e].
        G_W = torch.einsum("bnhd,bne->hde", G_Y, X).reshape(H * dh, X.shape[-1])
        return G_W.detach(), Y_chunks.detach(), mean_vcos

    def _delta_and_sig_stacked(
        self,
        Y_q: torch.Tensor,           # (B, N, H, dh)
        Y_k: torch.Tensor,
        Y_v: torch.Tensor,
        S_q_tgt: torch.Tensor,       # (B, H, n_chunks, value_chunk_dim)
        S_k_tgt: torch.Tensor,
        S_v_tgt: torch.Tensor,
        s_bank: torch.Tensor,        # (B,)
        X: torch.Tensor,             # (B, N, D)
        pad_mask: torch.Tensor,      # (B, n_chunks)
        seq_lengths: Optional[torch.Tensor],
        want_diagnostics: bool = True,
    ):
        """
        Same math as three _delta_and_sig calls, but Q/K/V stacked along
        the batch axis so a single encoder pass and a single autograd.grad
        traversal cover all three weight matrices. Only valid when the
        value encoders are stateless (center=False, the default): the
        encoder is then a pure pad+chunk reshape, identical across Q/K/V.

        Returns:
            (G_W_q, G_W_k, G_W_v): each (H*dh, D).
            (S_q_s, S_k_s, S_v_s): each (B, H, n_chunks, Dv), detached.
            (vcos_q, vcos_k, vcos_v): floats (0.0 off log steps).
        """
        B, N, H, dh = Y_q.shape
        C = self.key_encoder.n_chunks
        Lc = self.chunk_length
        Dv = self.value_encoder_q.chunk_dim

        # Token-level validity, shared across Q/K/V and heads.
        if seq_lengths is not None:
            token_valid = (
                torch.arange(N, device=X.device).unsqueeze(0)
                < seq_lengths.unsqueeze(1)
            ).to(X.dtype)                                       # (B, N)
        else:
            token_valid = None

        with torch.enable_grad():
            Y_leaf = torch.stack([Y_q, Y_k, Y_v], dim=0).detach()
            Y_leaf.requires_grad_(True)                         # (3, B, N, H, dh)
            Yz = Y_leaf
            if token_valid is not None:
                Yz = Yz * token_valid.view(1, B, N, 1, 1)
            # Right-pad the token axis to max_seq_length (zeros, matching
            # the encoder), then chunk-reshape. Stateless encoder == pure
            # pad+chunk: (3, B, Nm, H, dh) -> (3, B, H, C, Lc*dh).
            Nm = self.max_seq_length
            if N < Nm:
                Yz = torch.nn.functional.pad(
                    Yz, (0, 0, 0, 0, 0, Nm - N)
                )
            Y_chunks = (
                Yz.permute(0, 1, 3, 2, 4)                       # (3,B,H,Nm,dh)
                .reshape(3, B, H, C, Lc * dh)
            )
            S_all = torch.stack([S_q_tgt, S_k_tgt, S_v_tgt], dim=0)
            # (3, B, H, C, Dv)

            dot = (Y_chunks * S_all).sum(dim=-1)                # (3, B, H, C)
            ny = Y_chunks.norm(dim=-1).clamp_min(self.eps)
            nt = S_all.norm(dim=-1).clamp_min(self.eps)
            cos = dot / (ny * nt)                               # (3, B, H, C)

            valid = (~pad_mask).to(cos.dtype).view(1, B, 1, C)
            cos = cos * valid

            total = (cos * s_bank.view(1, B, 1, 1)).sum()
            (G_Y,) = torch.autograd.grad(total, Y_leaf, create_graph=False)
            # G_Y: (3, B, N, H, dh)

        vcos_q = vcos_k = vcos_v = 0.0
        if want_diagnostics:
            with torch.no_grad():
                n_valid = (valid.sum() * H).clamp_min(1.0)
                vcos = (cos.sum(dim=(1, 2, 3)) / n_valid).cpu()
                vcos_q, vcos_k, vcos_v = (float(v) for v in vcos)

        # Per-segment outer product with X: (3, H*dh, D).
        G_W = torch.einsum("sbnhd,bne->shde", G_Y, X).reshape(
            3, H * dh, X.shape[-1]
        )
        S_student = Y_chunks.detach()
        return (
            (G_W[0].detach(), G_W[1].detach(), G_W[2].detach()),
            (S_student[0], S_student[1], S_student[2]),
            (vcos_q, vcos_k, vcos_v),
        )

    @torch.no_grad()
    def _compute_separation_metrics(
        self,
        chunks: torch.Tensor,   # (B, n_chunks, key_chunk_dim)
        j_star: torch.Tensor,   # (B,)
        labels: torch.Tensor,   # (B,)
    ) -> Dict[str, float]:
        """Delegates to the module-level separation_metrics() helper
        (shared with the cross-layer batched implementation)."""
        return separation_metrics(chunks, j_star, labels, eps=self.eps)

    def _clear_cache(self):
        self._cached_X = None
        self._cached_Y_q = None
        self._cached_Y_k = None
        self._cached_Y_v = None
        self._cached_seq_lengths = None
        self._cached_labels = None

    def extra_repr(self):
        return (
            f"D={self.input_dim}, H={self.num_heads}, dh={self.d_head}, "
            f"N_max={self.max_seq_length}, L={self.chunk_length}, "
            f"alpha={self.alpha}, warmup={self.warmup_steps}"
        )