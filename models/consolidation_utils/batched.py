"""
StackedConsolidation: cross-layer batched hippocampal-indexed consolidation.

Functionally equivalent to L independent ConsolidationModules (one per
attention layer) but with every per-step tensor op batched over a leading
layer dimension: one key-encode, one retrieval einsum, one autograd graph
covering all layers AND all three of Q/K/V, one outer-product einsum, and
one set of scatter_adds for the bank EMA. This replaces ~12 sequential
small-kernel pipelines per training step with single large launches and
removes all per-step host-device syncs (diagnostics are materialized only
when want_diagnostics=True, i.e. on log steps).

Constraints relative to the per-layer path (all satisfied by the ViT
training setup; use ConsolidationCL(use_batched=False) otherwise):
  - all layers share one config (dims, chunking, slots, betas, alpha);
  - no seq_lengths / padding: every cached X must have N == max_seq_length;
  - value-side stats disabled (center_values=False, the default).

State is interconvertible with the per-layer format: per_layer_state_dicts()
emits {"cm_{i}": <ConsolidationModule-style state dict>} and
load_per_layer_state_dicts() accepts the same, so checkpoints written by
either implementation load into the other.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import SlotFeatureDecoder
from .module import separation_metrics
from .replay_ops import head_replay_step, histogram_disc, priority_score, slot_value_summary, summary_dim


class StackedConsolidation(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        num_heads: int,
        d_head: int,
        weights_q: List[torch.Tensor],
        weights_k: List[torch.Tensor],
        weights_v: List[torch.Tensor],
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
        decode_layer: int = -1,
        eps: float = 1e-6,
        enc_eps: float = 1e-4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if not (len(weights_q) == len(weights_k) == len(weights_v) == n_layers):
            raise ValueError("need one weight per layer for each of Q/K/V")
        if chunk_length is None:
            chunk_length = max_seq_length
        if max_seq_length % chunk_length != 0:
            raise ValueError(
                f"max_seq_length ({max_seq_length}) must be divisible by "
                f"chunk_length ({chunk_length})"
            )
        if top_k < 1 or top_k > n_slots:
            raise ValueError(f"top_k must be in [1, {n_slots}], got {top_k}")
        if top_k > 1 and retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be > 0 for top_k > 1")
        if not 0.0 <= blend_floor <= 1.0:
            raise ValueError(f"blend_floor must be in [0,1], got {blend_floor}")
        if dead_restart_interval < 0:
            raise ValueError(
                f"dead_restart_interval must be >= 0, got {dead_restart_interval}"
            )

        expected_w = (num_heads * d_head, input_dim)
        for ws in (weights_q, weights_k, weights_v):
            for W in ws:
                if tuple(W.shape) != expected_w:
                    raise ValueError(
                        f"weight shape {tuple(W.shape)} != expected {expected_w}"
                    )

        self.L = n_layers
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_head = d_head
        self.max_seq_length = max_seq_length
        self.chunk_length = chunk_length
        self.n_chunks = max_seq_length // chunk_length
        self.pad_mode = pad_mode
        self.J = n_slots
        self.beta_key = beta_key
        self.beta_value = beta_value
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.cache_projections = cache_projections
        self.center_keys = center_keys
        self.standardize_keys = standardize_keys
        self.center_momentum = center_momentum
        self.normalize_delta = normalize_delta
        self.projection_dim = projection_dim
        self.projection_train_steps = projection_train_steps
        self.projection_train_always = projection_train_always
        self.supcon_temperature = float(supcon_temperature)
        self.top_k = top_k
        self.retrieval_temperature = float(retrieval_temperature)
        self.blend_floor = float(blend_floor)
        self.data_init = bool(data_init)
        self.dead_restart_interval = int(dead_restart_interval)
        self.dead_min_hits = float(dead_min_hits)
        self.dead_protect_hits = float(dead_protect_hits)
        self.novelty_threshold = float(novelty_threshold)
        self.eps = eps
        self.enc_eps = enc_eps

        # External weight views, [Q, K, V] x layer. Updates via .data.add_().
        self._weights = [list(weights_q), list(weights_k), list(weights_v)]

        L, J, C, H = n_layers, n_slots, self.n_chunks, num_heads
        enc_dim = projection_dim if projection_dim is not None else input_dim
        key_chunk_dim = chunk_length * enc_dim
        value_chunk_dim = chunk_length * d_head
        self.key_chunk_dim = key_chunk_dim
        self.value_chunk_dim = value_chunk_dim

        # Bank keys: per-layer seeded exactly like the per-layer modules
        # (seed + layer_index), so both implementations initialize
        # identically for a given seed.
        keys = torch.empty(L, J, C, key_chunk_dim)
        for layer in range(L):
            if seed is not None:
                gen = torch.Generator()
                gen.manual_seed(seed + layer)
                keys[layer] = torch.randn(J, C, key_chunk_dim, generator=gen)
            else:
                keys[layer] = torch.randn(J, C, key_chunk_dim)
        self.register_buffer("keys", keys)
        for name in ("values_q", "values_k", "values_v"):
            self.register_buffer(
                name, torch.zeros(L, J, H, C, value_chunk_dim)
            )
        self.register_buffer("hit_count", torch.zeros(L, J))

        # Slot-recruitment state (see HippocampalBank for semantics).
        self.register_buffer(
            "keys_initialized",
            torch.full((L, J), not data_init, dtype=torch.bool),
        )
        self.register_buffer("hit_snapshot", torch.zeros(L, J))
        self.register_buffer(
            "updates_since_restart", torch.zeros((), dtype=torch.long)
        )
        self._all_seeded: bool = not data_init
        self._updates_py: int = 0

        # Slot-prioritized-replay state. Tracked for the single decode
        # layer only (the decoder reads that layer's values). n_classes==0
        # (default) allocates nothing and keeps the kernel byte-identical
        # to the pre-replay path.
        self.n_classes = int(n_classes)
        self.decode_layer = decode_layer if decode_layer >= 0 else n_layers + decode_layer
        if not 0 <= self.decode_layer < n_layers:
            raise ValueError(
                f"decode_layer {decode_layer} out of range for {n_layers} layers")
        if self.n_classes > 0:
            self.register_buffer(
                "slot_class_hist", torch.zeros(n_slots, self.n_classes))
            self.register_buffer("task_hit_count", torch.zeros(n_slots))
        # Decoder + replay scratch (head-mode); weight-mode re-ascends
        # _last_steps. All None until setup_decoder / first consolidate.
        self.decoder: Optional[SlotFeatureDecoder] = None
        self.decoder_optimizer: Optional[torch.optim.Optimizer] = None
        self.decoder_val_dim: Optional[int] = None
        self.decoder_summary_mode: str = "mean"
        self._last_j_star: Optional[torch.Tensor] = None
        self._last_steps: Optional[torch.Tensor] = None

        if center_keys:
            self.register_buffer("key_mean", torch.zeros(L, enc_dim))
            self.register_buffer("key_var", torch.ones(L, enc_dim))
            self.register_buffer(
                "stats_initialized", torch.zeros((), dtype=torch.bool)
            )

        # Learned per-layer projection MLPs, stored as stacked parameters
        # so both SupCon training and the key-side forward run as batched
        # einsums. Initialized exactly like L independent nn.Linear pairs.
        if projection_dim is not None:
            w1 = torch.empty(L, projection_hidden, input_dim)
            b1 = torch.empty(L, projection_hidden)
            w2 = torch.empty(L, projection_dim, projection_hidden)
            b2 = torch.empty(L, projection_dim)
            for layer in range(L):
                lin1 = nn.Linear(input_dim, projection_hidden)
                lin2 = nn.Linear(projection_hidden, projection_dim)
                w1[layer] = lin1.weight.detach()
                b1[layer] = lin1.bias.detach()
                w2[layer] = lin2.weight.detach()
                b2[layer] = lin2.bias.detach()
            self.proj_w1 = nn.Parameter(w1)
            self.proj_b1 = nn.Parameter(b1)
            self.proj_w2 = nn.Parameter(w2)
            self.proj_b2 = nn.Parameter(b2)
        else:
            self.proj_w1 = self.proj_b1 = self.proj_w2 = self.proj_b2 = None

        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))
        self._step_py: int = 0
        self.register_load_state_dict_post_hook(self._sync_step_py)

        self._cached_X: List[Optional[torch.Tensor]] = [None] * L
        self._cached_Y: List[List[Optional[torch.Tensor]]] = [
            [None] * L for _ in range(3)
        ]
        self._cached_labels: Optional[torch.Tensor] = None

    @staticmethod
    def _sync_step_py(module, incompatible_keys):
        module._step_py = int(module.step_count.item())
        module._all_seeded = bool(module.keys_initialized.all().item())
        module._updates_py = int(module.updates_since_restart.item())

    @torch.no_grad()
    def _seed_slots(
        self,
        layer: int,
        slot_idx: torch.Tensor,
        donor_chunks: torch.Tensor,
        reset_values: bool = True,
    ) -> None:
        """Overwrite keys[layer, slot_idx] with donor chunks (+1% noise)."""
        if slot_idx.numel() == 0:
            return
        noise = 0.01 * donor_chunks.std().clamp_min(self.eps)
        self.keys[layer, slot_idx] = (
            donor_chunks + noise * torch.randn_like(donor_chunks)
        )
        if reset_values:
            for buf in (self.values_q, self.values_k, self.values_v):
                buf[layer, slot_idx] = 0.0
        # Clear replay tags for recycled decode-layer slots (see
        # HippocampalBank._seed_slots).
        if self.n_classes > 0 and layer == self.decode_layer:
            self.slot_class_hist[slot_idx] = 0.0
            self.task_hit_count[slot_idx] = 0.0
        self.keys_initialized[layer, slot_idx] = True

    @property
    def projection_params(self) -> List[nn.Parameter]:
        if self.proj_w1 is None:
            return []
        return [self.proj_w1, self.proj_b1, self.proj_w2, self.proj_b2]

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache(
        self,
        layer: int,
        X: torch.Tensor,
        Y_q: Optional[torch.Tensor] = None,
        Y_k: Optional[torch.Tensor] = None,
        Y_v: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        self._cached_X[layer] = X.detach()
        if self.cache_projections:
            if Y_q is None or Y_k is None or Y_v is None:
                raise ValueError("cache_projections=True requires Y_q/Y_k/Y_v")
            self._cached_Y[0][layer] = Y_q.detach()
            self._cached_Y[1][layer] = Y_k.detach()
            self._cached_Y[2][layer] = Y_v.detach()
        if labels is not None:
            self._cached_labels = labels.detach()

    def _clear_cache(self):
        self._cached_X = [None] * self.L
        self._cached_Y = [[None] * self.L for _ in range(3)]
        self._cached_labels = None

    def on_task_start(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Shared sub-computations
    # ------------------------------------------------------------------

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        """Per-layer MLP via batched einsum. X: (L, B, N, D) -> (L, B, N, P)."""
        h = torch.einsum("lbnd,lhd->lbnh", X, self.proj_w1)
        h = h + self.proj_b1.view(self.L, 1, 1, -1)
        h = F.gelu(h)
        p = torch.einsum("lbnh,lph->lbnp", h, self.proj_w2)
        return p + self.proj_b2.view(self.L, 1, 1, -1)

    def _encode_keys(
        self, X: torch.Tensor, update_stats: bool
    ) -> torch.Tensor:
        """Center/standardize (per-layer running stats) + chunk.

        X: (L, B, N, E) -> chunks (L, B, C, chunk_length*E).
        """
        L, B, N, E = X.shape
        if self.center_keys:
            if update_stats and self.training:
                flat = X.reshape(L, B * N, E)
                batch_mean = flat.mean(dim=1)                  # (L, E)
                batch_var = flat.var(dim=1, unbiased=False)    # (L, E)
                if not bool(self.stats_initialized):
                    self.key_mean.copy_(batch_mean)
                    self.key_var.copy_(batch_var)
                    self.stats_initialized.fill_(True)
                else:
                    m = self.center_momentum
                    self.key_mean.mul_(m).add_(batch_mean, alpha=1 - m)
                    self.key_var.mul_(m).add_(batch_var, alpha=1 - m)
            X = X - self.key_mean.view(L, 1, 1, E)
            if self.standardize_keys:
                X = X / (self.key_var.view(L, 1, 1, E) + self.enc_eps).sqrt()
        return X.view(L, B, self.n_chunks, self.chunk_length * E)

    def _retrieve(self, chunks: torch.Tensor):
        """Batched bank retrieval.

        chunks: (L, B, C, Dk). Returns s (L,B), j_top (L,B,k), w_top (L,B,k),
        top_sims (L,B,k), and blended targets S_q/S_k/S_v (L,B,H,C,Dv).
        """
        L, B = chunks.shape[:2]
        cn = F.normalize(chunks, dim=-1, eps=self.eps)
        kn = F.normalize(self.keys, dim=-1, eps=self.eps)
        sims = torch.einsum("lbcd,ljcd->lbjc", cn, kn).mean(dim=-1)  # (L,B,J)

        if self.top_k == 1:
            s, j_star = sims.max(dim=-1)                       # (L, B)
            j_top = j_star.unsqueeze(-1)
            top_sims = s.unsqueeze(-1)
            w_top = torch.ones_like(top_sims)
            l_idx = torch.arange(L, device=chunks.device).view(L, 1)
            S_q = self.values_q[l_idx, j_star]                 # (L,B,H,C,Dv)
            S_k = self.values_k[l_idx, j_star]
            S_v = self.values_v[l_idx, j_star]
        else:
            k = self.top_k
            top_sims, j_top = sims.topk(k, dim=-1)             # (L, B, k)
            w_top = F.softmax(top_sims / self.retrieval_temperature, dim=-1)
            s = (w_top * top_sims).sum(dim=-1)                 # (L, B)
            H, C, Dv = self.num_heads, self.n_chunks, self.value_chunk_dim
            S_q = torch.zeros(L, B, H, C, Dv,
                              dtype=chunks.dtype, device=chunks.device)
            S_k = torch.zeros_like(S_q)
            S_v = torch.zeros_like(S_q)
            l_idx = torch.arange(L, device=chunks.device).view(L, 1)
            for r in range(k):
                j_r = j_top[:, :, r]                           # (L, B)
                w_r = w_top[:, :, r].view(L, B, 1, 1, 1)
                S_q += self.values_q[l_idx, j_r] * w_r
                S_k += self.values_k[l_idx, j_r] * w_r
                S_v += self.values_v[l_idx, j_r] * w_r
        return s, j_top, w_top, top_sims, S_q, S_k, S_v

    def _bank_update(
        self,
        j_top: torch.Tensor,        # (L, B, k)
        w_top: torch.Tensor,        # (L, B, k)
        chunks: torch.Tensor,       # (L, B, C, Dk)
        S_students: torch.Tensor,   # (3, L, B, H, C, Dv)
        top_sims: Optional[torch.Tensor] = None,  # (L, B, k)
        labels: Optional[torch.Tensor] = None,    # (B,)
    ) -> None:
        """Weight-scaled EMA update, batched over layers.

        Same semantics as HippocampalBank.update (with the corrected
        weight-scaled blend factor): per (layer, slot, chunk),
            new = old + (1-beta) * max(blend_floor, min(total_w, 1)) * (mean - old),
        plus the same slot-recruitment mechanisms (data init, novelty
        claiming, dead-slot restarts).
        """
        L, B, k = j_top.shape
        C, J = self.n_chunks, self.J
        H, Dv, Dk = self.num_heads, self.value_chunk_dim, self.key_chunk_dim
        dev, dt = chunks.device, chunks.dtype

        # ----- Data-dependent key init (see HippocampalBank.update).
        if self.data_init and not self._all_seeded:
            for layer in range(L):
                uninit = (~self.keys_initialized[layer]).nonzero(as_tuple=True)[0]
                if uninit.numel() == 0:
                    continue
                donors = torch.randperm(B, device=dev)[:uninit.numel()]
                self._seed_slots(layer, uninit[:donors.numel()],
                                 chunks[layer, donors], reset_values=False)
            self._all_seeded = bool(self.keys_initialized.all().item())

        # Global slot index: layer * J + slot, so one scatter covers all
        # layers.
        base = (torch.arange(L, device=dev) * J).view(L, 1)    # (L, 1)

        # ----- Total weight per (layer*slot, chunk) -----
        count = torch.zeros(L * J, C, dtype=dt, device=dev)
        for r in range(k):
            g_r = (base + j_top[:, :, r]).reshape(L * B)       # (L*B,)
            w_r = w_top[:, :, r].reshape(L * B, 1).expand(L * B, C)
            count.scatter_add_(0, g_r.view(L * B, 1).expand(L * B, C), w_r)
        touched = count > self.eps
        denom = count.clamp_min(self.eps)
        m_eff = torch.where(
            touched,
            count.clamp(min=self.blend_floor, max=1.0),
            torch.zeros_like(count),
        )                                                      # (L*J, C)

        # ----- Keys -----
        sum_keys = torch.zeros(L * J, C, Dk, dtype=dt, device=dev)
        chunks_flat = chunks.reshape(L * B, C, Dk)
        for r in range(k):
            g_r = (base + j_top[:, :, r]).reshape(L * B)
            w_r = w_top[:, :, r].reshape(L * B, 1, 1)
            idx = g_r.view(L * B, 1, 1).expand(L * B, C, Dk)
            sum_keys.scatter_add_(0, idx, chunks_flat * w_r)
        mean_keys = sum_keys / denom.unsqueeze(-1)
        keys_flat = self.keys.view(L * J, C, Dk)
        keys_flat.add_(
            (1 - self.beta_key) * m_eff.unsqueeze(-1) * (mean_keys - keys_flat)
        )

        # ----- Values (Q, K, V) -----
        m_v = m_eff.view(L * J, 1, C, 1)
        denom_v = denom.view(L * J, 1, C, 1)
        bv = self.beta_value
        for si, buf in enumerate((self.values_q, self.values_k, self.values_v)):
            S_flat = S_students[si].reshape(L * B, H, C, Dv)
            sum_buf = torch.zeros(L * J, H, C, Dv, dtype=dt, device=dev)
            for r in range(k):
                g_r = (base + j_top[:, :, r]).reshape(L * B)
                w_r = w_top[:, :, r].reshape(L * B, 1, 1, 1)
                idx = g_r.view(L * B, 1, 1, 1).expand(L * B, H, C, Dv)
                sum_buf.scatter_add_(0, idx, S_flat * w_r)
            mean_buf = sum_buf / denom_v
            buf_flat = buf.view(L * J, H, C, Dv)
            buf_flat.add_((1 - bv) * m_v * (mean_buf - buf_flat))

        # ----- Hit count -----
        hit_flat = self.hit_count.view(L * J)
        for r in range(k):
            g_r = (base + j_top[:, :, r]).reshape(L * B)
            hit_flat.scatter_add_(0, g_r, w_top[:, :, r].reshape(L * B).to(dt))

        # ----- Replay bookkeeping (decode layer, top-1 slot only) -----
        if self.n_classes > 0 and labels is not None:
            j0 = j_top[self.decode_layer, :, 0]                # (B,) top-1 slot
            lbl = labels.reshape(-1).to(torch.long)
            if lbl.shape[0] == B:
                ones = torch.ones(B, device=dev, dtype=self.slot_class_hist.dtype)
                flat_idx = j0 * self.n_classes + lbl
                self.slot_class_hist.view(-1).scatter_add_(0, flat_idx, ones)
                self.task_hit_count.scatter_add_(
                    0, j0, ones.to(self.task_hit_count.dtype))

        self.updates_since_restart += 1
        self._updates_py += 1

        # ----- Novelty-triggered claiming (one host sync when enabled).
        if self.novelty_threshold > -1.0 and top_sims is not None:
            novel = top_sims[:, :, 0] < self.novelty_threshold  # (L, B)
            n_novel_cpu = novel.sum(dim=1).cpu()
            for layer in range(L):
                n_novel = int(n_novel_cpu[layer])
                if n_novel == 0:
                    continue
                claim = torch.argsort(self.hit_count[layer])[:n_novel]
                self._seed_slots(layer, claim, chunks[layer, novel[layer]])
                self.hit_count[layer, claim] += 1.0

        # ----- Dead-slot restart (every dead_restart_interval steps).
        if (self.dead_restart_interval > 0
                and self._updates_py >= self.dead_restart_interval):
            for layer in range(L):
                gained = self.hit_count[layer] - self.hit_snapshot[layer]
                dead = ((gained < self.dead_min_hits)
                        & (self.hit_count[layer] < self.dead_protect_hits)
                        ).nonzero(as_tuple=True)[0]
                if dead.numel() > 0:
                    donors = torch.randint(0, B, (dead.numel(),), device=dev)
                    self._seed_slots(layer, dead, chunks[layer, donors])
            self.hit_snapshot.copy_(self.hit_count)
            self.updates_since_restart.zero_()
            self._updates_py = 0

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def consolidate_all(
        self,
        lr_now: Optional[float] = None,
        lr_anchor: Optional[float] = None,
        compute_separation_metrics: bool = False,
        want_diagnostics: bool = True,
    ) -> List[Dict[str, object]]:
        """One consolidation step for all layers at once.

        Returns a list of per-layer diagnostics dicts (same keys as
        ConsolidationModule.consolidate) when want_diagnostics, else [].
        """
        if any(x is None for x in self._cached_X):
            missing = [i for i, x in enumerate(self._cached_X) if x is None]
            raise RuntimeError(
                f"No cached input for layers {missing}. Call cache() for "
                f"every layer before consolidate_all()."
            )

        dt = self.keys.dtype
        X = torch.stack(self._cached_X).to(dtype=dt)           # (L, B, N, D)
        L, B, N, D = X.shape
        if N != self.max_seq_length:
            raise RuntimeError(
                f"Batched consolidation requires N == max_seq_length "
                f"({self.max_seq_length}), got {N}. Use the per-layer path "
                f"for variable-length input."
            )
        if D != self.input_dim:
            raise RuntimeError(f"X feature dim {D} != input_dim {self.input_dim}")
        labels = self._cached_labels
        H, dh = self.num_heads, self.d_head
        C, Lc = self.n_chunks, self.chunk_length
        O = H * dh

        # 1. Resolve Y: cached projections, or recompute from the current
        # (post-optimizer-step) weights.
        if self.cache_projections:
            Y_all = torch.stack([
                torch.stack(self._cached_Y[si]).to(dtype=dt)
                for si in range(3)
            ])                                                  # (3, L, B, N, O)
        else:
            W_all = torch.stack([
                torch.stack([w.detach().to(dtype=dt) for w in self._weights[si]])
                for si in range(3)
            ])                                                  # (3, L, O, D)
            Y_all = torch.einsum("lbnd,slod->slbno", X, W_all)

        # 2. Key encode (projection -> running stats -> chunk).
        X_for_keys = self._project(X) if self.proj_w1 is not None else X
        chunks = self._encode_keys(X_for_keys, update_stats=True)
        # (L, B, C, Dk)

        # Cleared each call; set only when a step is applied (see weight_replay).
        self._last_steps = None

        # 3. Retrieve.
        s, j_top, w_top, top_sims, S_q, S_k, S_v = self._retrieve(chunks)
        S_all = torch.stack([S_q, S_k, S_v])                    # (3,L,B,H,C,Dv)
        # Decode-layer top-1 slot for replay anchoring (no host sync).
        self._last_j_star = j_top[self.decode_layer, :, 0].detach()

        # 4. One autograd graph: per-(s,l,b,h,c) cosine vs target, gradient
        # w.r.t. all Y at once, then per-(s,l) outer product with X.
        with torch.enable_grad():
            Y_leaf = Y_all.detach().requires_grad_(True)        # (3,L,B,N,O)
            Y_chunks = (
                Y_leaf.view(3, L, B, N, H, dh)
                .permute(0, 1, 2, 4, 3, 5)                      # (3,L,B,H,N,dh)
                .reshape(3, L, B, H, C, Lc * dh)
            )
            dot = (Y_chunks * S_all).sum(dim=-1)                # (3,L,B,H,C)
            ny = Y_chunks.norm(dim=-1).clamp_min(self.eps)
            nt = S_all.norm(dim=-1).clamp_min(self.eps)
            cos = dot / (ny * nt)
            total = (cos * s.view(1, L, B, 1, 1)).sum()
            (G_Y,) = torch.autograd.grad(total, Y_leaf, create_graph=False)

        G_W = torch.einsum("slbno,lbnd->slod", G_Y, X)          # (3, L, O, D)
        norms = G_W.reshape(3, L, -1).norm(dim=-1)              # (3, L)

        # 5. Bank EMA update with the student outputs.
        self._bank_update(j_top, w_top, chunks, Y_chunks.detach(),
                          top_sims=top_sims, labels=labels)

        # 6. Apply weight update (warmup-gated; python-side counter so no
        # GPU sync off log steps).
        self.step_count += 1
        self._step_py += 1
        in_warmup = self._step_py <= self.warmup_steps

        alpha_eff = self.alpha
        lr_ratio = 1.0
        if not in_warmup and lr_now is not None and lr_anchor is not None:
            lr_ratio = float(lr_now) / max(float(lr_anchor), 1e-12)
            alpha_eff = self.alpha * lr_ratio

        if not in_warmup and alpha_eff > 0:
            if self.normalize_delta:
                steps = G_W / (norms.view(3, L, 1, 1) + self.eps)
            else:
                steps = G_W
            for si in range(3):
                for layer in range(L):
                    self._weights[si][layer].data.add_(
                        alpha_eff * steps[si, layer]
                    )
            # Stash for weight-mode replay re-ascension at a floored alpha.
            self._last_steps = steps.detach()

        if not want_diagnostics:
            self._clear_cache()
            return []

        # ---- Diagnostics (log steps only; this is where syncs happen) ----
        k = self.top_k
        s_mean = s.mean(dim=1).cpu()
        s_min = s.min(dim=1).values.cpu()
        s_max = s.max(dim=1).values.cpu()
        j_star = j_top[:, :, 0]                                 # (L, B)
        norms_cpu = norms.cpu()
        vcos = cos.mean(dim=(2, 3, 4)).cpu()                    # (3, L)
        top_sims_mean = top_sims.mean(dim=1).cpu()              # (L, k)
        w_top_mean = w_top.mean(dim=1).cpu()                    # (L, k)
        if k > 1:
            w_safe = w_top.clamp_min(self.eps)
            ent = -(w_safe * w_safe.log()).sum(dim=-1)          # (L, B)
            ent_mean = ent.mean(dim=1).cpu()
            ent_max = ent.max(dim=1).values.cpu()
            ent_min = ent.min(dim=1).values.cpu()
        j_star_cpu = j_star.cpu()
        j_top_cpu = j_top.cpu()

        diags: List[Dict[str, object]] = []
        for layer in range(L):
            d: Dict[str, object] = {
                "step": self._step_py,
                "in_warmup": bool(in_warmup),
                "s_mean": float(s_mean[layer]),
                "s_min": float(s_min[layer]),
                "s_max": float(s_max[layer]),
                "j_star": j_star_cpu[layer],
                "delta_norm_q": float(norms_cpu[0, layer]),
                "delta_norm_k": float(norms_cpu[1, layer]),
                "delta_norm_v": float(norms_cpu[2, layer]),
                "alpha_eff": float(alpha_eff),
                "lr_ratio": float(lr_ratio),
                "vcos_q": float(vcos[0, layer]),
                "vcos_k": float(vcos[1, layer]),
                "vcos_v": float(vcos[2, layer]),
                "unique_slots_hit": int(j_star_cpu[layer].unique().numel()),
                "top_k": k,
                "unique_topk_slots": int(j_top_cpu[layer].unique().numel()),
                "top_sims_mean_per_rank": top_sims_mean[layer].tolist(),
                "w_top_mean_per_rank": w_top_mean[layer].tolist(),
                "w_top_entropy_mean": float(ent_mean[layer]) if k > 1 else 0.0,
                "w_top_entropy_max": float(ent_max[layer]) if k > 1 else 0.0,
                "w_top_entropy_min": float(ent_min[layer]) if k > 1 else 0.0,
            }
            if compute_separation_metrics and labels is not None and B >= 2:
                d.update(separation_metrics(
                    chunks[layer], j_star[layer], labels, eps=self.eps,
                ))
            diags.append(d)

        self._clear_cache()
        return diags

    # ------------------------------------------------------------------
    # Slot-prioritized replay (decode-layer slice; mirrors the per-layer
    # ConsolidationModule API so the wrapper treats either backend the same)
    # ------------------------------------------------------------------

    def setup_decoder(
        self,
        feature_dim: Optional[int] = None,
        hidden: int = 384,
        noise: float = 0.0,
        lr: float = 1e-3,
        summary_mode: str = "mean",
    ) -> SlotFeatureDecoder:
        if feature_dim is None:
            feature_dim = self.input_dim
        self.decoder_summary_mode = summary_mode
        self.decoder_val_dim = summary_dim(self.num_heads, self.d_head, summary_mode)
        self.decoder = SlotFeatureDecoder(
            self.decoder_val_dim, feature_dim, hidden=hidden, noise=noise,
        ).to(self.keys.device)
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(), lr=lr, weight_decay=0.0)
        return self.decoder

    @torch.no_grad()
    def reset_task_hits(self) -> None:
        if self.n_classes > 0:
            self.task_hit_count.zero_()

    @torch.no_grad()
    def replay_stats(self):
        """(T_j, P_j, disc_j, gate) for the decode layer (see
        HippocampalBank.replay_stats)."""
        if self.n_classes <= 0:
            raise RuntimeError(
                "replay_stats() requires n_classes > 0 (replay enabled).")
        hc = self.hit_count[self.decode_layer]
        T = self.task_hit_count.clone()
        P = (hc - self.task_hit_count).clamp_min(0.0)
        disc = histogram_disc(self.slot_class_hist, self.n_classes, self.eps)
        gate = (hc > 0) & self.keys_initialized[self.decode_layer]
        return T, P, disc, gate

    @torch.no_grad()
    def replay_priority(self, w_single, w_cross, min_usage):
        T, P, disc, gate = self.replay_stats()
        return priority_score(T, P, disc, gate, self.hit_count[self.decode_layer],
                              w_single, w_cross, min_usage)

    @torch.no_grad()
    def slot_value_matrix(self) -> torch.Tensor:
        dl = self.decode_layer
        return slot_value_summary(
            self.values_q[dl], self.values_k[dl], self.values_v[dl],
            self.num_heads, self.d_head, self.n_chunks, self.chunk_length,
            mode=self.decoder_summary_mode)

    def replay_step(
        self, head, loss_fn, m, replay_weight, anchor_weight,
        w_single, w_cross, min_usage, enable_ce,
        anchor_j_star=None, anchor_feats=None, head_optimizer=None,
        want_diagnostics=False,
    ) -> Dict[str, object]:
        """Head-mode replay on the decode layer (delegates to the shared
        head_replay_step, identical to the per-layer path)."""
        if self.decoder is None:
            raise RuntimeError("setup_decoder() must run before replay_step()")
        return head_replay_step(
            decoder=self.decoder, decoder_optimizer=self.decoder_optimizer,
            summaries=self.slot_value_matrix(),
            slot_class_hist=self.slot_class_hist,
            priority=self.replay_priority(w_single, w_cross, min_usage),
            head=head, loss_fn=loss_fn, m=m,
            replay_weight=replay_weight, anchor_weight=anchor_weight,
            enable_ce=enable_ce,
            anchor_j_star=anchor_j_star, anchor_feats=anchor_feats,
            head_optimizer=head_optimizer, eps=self.eps,
            want_diagnostics=want_diagnostics)

    @torch.no_grad()
    def weight_replay(self, alpha_floor: float) -> int:
        """Re-ascend the last consolidation step (all layers) at a floored,
        schedule-independent alpha. Returns the number of layers applied."""
        if self._last_steps is None or alpha_floor <= 0.0:
            return 0
        for si in range(3):
            for layer in range(self.L):
                self._weights[si][layer].data.add_(
                    alpha_floor * self._last_steps[si, layer])
        return self.L

    # ------------------------------------------------------------------
    # SupCon projection training
    # ------------------------------------------------------------------

    def projection_is_training_now(self) -> bool:
        if self.proj_w1 is None:
            return False
        if self.projection_train_always:
            return True
        return self._step_py < self.projection_train_steps

    def compute_supcon_loss(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Supervised contrastive loss on the projected keys, all layers in
        one batched graph. Mirrors ConsolidationModule.compute_supcon_loss
        per layer; the returned scalar is the mean over layers (matching
        ConsolidationCL's old sum/len of per-layer losses).

        Returns (loss_scalar, per_layer_losses (L,) detached), or
        (None, None) when inactive.
        """
        if not self.projection_is_training_now():
            return None, None
        if any(x is None for x in self._cached_X) or self._cached_labels is None:
            return None, None

        dt = self.keys.dtype
        X = torch.stack([x.detach() for x in self._cached_X]).to(dtype=dt)
        L, B, N, D = X.shape
        labels = self._cached_labels
        if B < 2:
            return None, None

        # Project with grad to the stacked params; running stats are
        # applied but NOT updated (the consolidate pass owns the update).
        Xp = self._project(X)                                   # (L, B, N, P)
        chunks = self._encode_keys(Xp, update_stats=False)      # (L, B, C, Dk)

        chunks_n = F.normalize(chunks, dim=-1, eps=self.eps)
        key = chunks_n.mean(dim=2)                              # (L, B, Dk)
        key = F.normalize(key, dim=-1, eps=self.eps)

        logits = torch.bmm(key, key.transpose(1, 2)) / self.supcon_temperature
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye.unsqueeze(0), float("-inf"))

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        same_class = same_class & ~eye                          # (B, B)

        logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)                          # (L, B, B)
        denom_p = exp_logits.sum(dim=-1).clamp_min(self.eps)    # (L, B)
        pos_exp = (
            exp_logits * same_class.to(exp_logits.dtype).unsqueeze(0)
        ).sum(dim=-1)                                           # (L, B)
        has_pos = same_class.any(dim=-1)                        # (B,)
        if not has_pos.any():
            return None, None
        per_row = -torch.log(pos_exp.clamp_min(self.eps) / denom_p)
        per_layer = per_row[:, has_pos].mean(dim=1)             # (L,)
        loss = per_layer.mean()
        return loss, per_layer.detach()

    # ------------------------------------------------------------------
    # Per-layer state dict interop
    # ------------------------------------------------------------------

    def per_layer_state_dicts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Export state in the per-layer ConsolidationModule format
        ({"cm_{i}": {...}}), so checkpoints stay loadable by either
        implementation (and by older analysis scripts)."""
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for layer in range(self.L):
            sd: Dict[str, torch.Tensor] = {
                "step_count": self.step_count.clone(),
                "bank.keys": self.keys[layer].clone(),
                "bank.values_q": self.values_q[layer].clone(),
                "bank.values_k": self.values_k[layer].clone(),
                "bank.values_v": self.values_v[layer].clone(),
                "bank.hit_count": self.hit_count[layer].clone(),
                "bank.keys_initialized": self.keys_initialized[layer].clone(),
                "bank.hit_snapshot": self.hit_snapshot[layer].clone(),
                "bank.updates_since_restart": self.updates_since_restart.clone(),
            }
            if self.center_keys:
                sd["key_encoder.running_mean"] = self.key_mean[layer].clone()
                sd["key_encoder.running_var"] = self.key_var[layer].clone()
                sd["key_encoder.stats_initialized"] = self.stats_initialized.clone()
            if self.proj_w1 is not None:
                sd["projection.0.weight"] = self.proj_w1[layer].detach().clone()
                sd["projection.0.bias"] = self.proj_b1[layer].detach().clone()
                sd["projection.2.weight"] = self.proj_w2[layer].detach().clone()
                sd["projection.2.bias"] = self.proj_b2[layer].detach().clone()
            out[f"cm_{layer}"] = sd
        return out

    def load_per_layer_state_dicts(
        self, sd: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """Load per-layer-format state ({"cm_{i}": {...}}), as written by
        the old per-layer ConsolidationCL or by per_layer_state_dicts()."""
        with torch.no_grad():
            for layer in range(self.L):
                key = f"cm_{layer}"
                if key not in sd:
                    continue
                m = sd[key]
                self.keys[layer].copy_(m["bank.keys"])
                self.values_q[layer].copy_(m["bank.values_q"])
                self.values_k[layer].copy_(m["bank.values_k"])
                self.values_v[layer].copy_(m["bank.values_v"])
                # Old checkpoints stored hit_count as long; accept both.
                self.hit_count[layer].copy_(m["bank.hit_count"].float())
                # Recruitment buffers; absent in checkpoints written before
                # the slot-recruitment mechanisms existed.
                if "bank.keys_initialized" in m:
                    self.keys_initialized[layer].copy_(m["bank.keys_initialized"])
                else:
                    self.keys_initialized[layer].fill_(True)
                if "bank.hit_snapshot" in m:
                    self.hit_snapshot[layer].copy_(m["bank.hit_snapshot"])
                if "bank.updates_since_restart" in m:
                    self.updates_since_restart.copy_(m["bank.updates_since_restart"])
                if self.center_keys and "key_encoder.running_mean" in m:
                    self.key_mean[layer].copy_(m["key_encoder.running_mean"])
                    self.key_var[layer].copy_(m["key_encoder.running_var"])
                    self.stats_initialized.copy_(
                        m["key_encoder.stats_initialized"]
                    )
                if self.proj_w1 is not None and "projection.0.weight" in m:
                    self.proj_w1[layer].copy_(m["projection.0.weight"])
                    self.proj_b1[layer].copy_(m["projection.0.bias"])
                    self.proj_w2[layer].copy_(m["projection.2.weight"])
                    self.proj_b2[layer].copy_(m["projection.2.bias"])
                if "step_count" in m:
                    self.step_count.copy_(m["step_count"])
        self._step_py = int(self.step_count.item())
        self._all_seeded = bool(self.keys_initialized.all().item())
        self._updates_py = int(self.updates_since_restart.item())

    def extra_repr(self):
        return (
            f"L={self.L}, D={self.input_dim}, H={self.num_heads}, "
            f"dh={self.d_head}, N_max={self.max_seq_length}, "
            f"Lc={self.chunk_length}, J={self.J}, alpha={self.alpha}, "
            f"warmup={self.warmup_steps}, top_k={self.top_k}"
        )
