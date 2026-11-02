"""
Hippocampal-indexed consolidation on ViT attention, as a Mammoth model.

Wraps the (unmodified) `models/consolidation_utils` package around any
ViT backbone with fused-qkv attention blocks (`backbone/vit.py` /
`backbone/vit_small.py`). No split-QKV surgery is needed:

  - A forward hook on each block's fused `attn.qkv` Linear receives the
    block input X (B, N, D) and the fused output (B, N, 3D), which is
    sliced into Y_q / Y_k / Y_v for the consolidation cache.
  - The consolidation weight updates are bound to row-slices of the
    fused weight: `qkv.weight.data[:D]`, `[D:2D]`, `[2D:]` are views
    into the same storage, so the module's in-place `.data.add_()`
    writes straight into the live parameter.

Binding happens lazily in `begin_task` of the first task, AFTER Mammoth
has moved the backbone to its device (`.to()` re-allocates parameter
storage, which would orphan views taken at construction time).

Per training step (`observe`):
  1. stash labels, forward (hooks fill the consolidation cache),
  2. backward + opt.step(),
  3. one SupCon step on the learned key projections (reads the cache),
  4. consolidate: bank retrieve -> cosine-target delta -> weight update.

The method is task-free: `begin_task`/`end_task` carry no method state
beyond lazy construction and error reporting.

NOTE on checkpoints: because the consolidation state is created lazily,
`--savecheck` checkpoints written after task 0 contain the bank buffers,
but loading them requires the model to have been through `begin_task`
once (or use `consolidation_utils` state dicts directly).
"""

import logging
import math

import torch

from models.utils.continual_model import ContinualModel
from models.consolidation_utils import ConsolidationModule, StackedConsolidation
from utils.args import ArgumentParser, binary_to_boolean_type


class ConsolidationVit(ContinualModel):
    """Hippocampal-indexed consolidation of attention Q/K/V weights."""

    NAME = 'consolidation_vit'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Defaults matching the reference bespoke run
        # (run_consolidation_topk3.sbatch): AdamW, lr 1e-4, wd 5e-4.
        parser.set_defaults(optimizer='adamw', lr=1e-4, optim_wd=5e-4)

        parser.add_argument('--consol_alpha', type=float, default=5e-4,
                            help='Consolidation step size (Frobenius step when normalize_delta).')
        parser.add_argument('--consol_normalize_delta', type=binary_to_boolean_type, default=1,
                            help='Unit-normalize each per-W delta before applying alpha.')
        parser.add_argument('--consol_n_slots', type=int, default=64,
                            help='Bank slots per layer.')
        parser.add_argument('--consol_beta_key', type=float, default=0.99,
                            help='EMA momentum for bank keys.')
        parser.add_argument('--consol_beta_value', type=float, default=0.999,
                            help='EMA momentum for bank values.')
        parser.add_argument('--consol_pad_mode', type=str, default='neg_one',
                            choices=['neg_one', 'zero', 'compute'],
                            help='Key-side treatment of padded chunks (moot for fixed-length ViT).')
        parser.add_argument('--consol_warmup_epochs', type=float, default=20.0,
                            help='Bank warmup in epochs (converted to steps at task 0; '
                                 'no weight updates during warmup).')
        parser.add_argument('--consol_warmup_steps', type=int, default=-1,
                            help='Bank warmup in steps; overrides --consol_warmup_epochs if >= 0.')
        parser.add_argument('--consol_chunk_length', type=int, default=-1,
                            help='Tokens per chunk; -1 = single chunk over the full sequence.')
        parser.add_argument('--consol_max_seq_length', type=int, default=-1,
                            help='Pad the token sequence up to this length before chunking; '
                                 '-1 = use the backbone sequence length (no extra padding). '
                                 'Must be >= the real sequence length and divisible by '
                                 '--consol_chunk_length. Use a composite value (e.g. 256) to '
                                 'enable chunk counts the prime backbone length (197) forbids.')
        parser.add_argument('--consol_projection_dim', type=int, default=128,
                            help='Learned key-projection output dim; -1 disables the projection.')
        parser.add_argument('--consol_projection_hidden', type=int, default=128,
                            help='Key-projection MLP hidden dim.')
        parser.add_argument('--consol_projection_lr', type=float, default=1e-3,
                            help='AdamW lr for the key-projection (SupCon) optimizer.')
        parser.add_argument('--consol_projection_train_always', type=binary_to_boolean_type, default=1,
                            help='Train the key projection for the whole run (reference setting).')
        parser.add_argument('--consol_projection_train_steps', type=int, default=0,
                            help='If not training always, number of steps to train the projection.')
        parser.add_argument('--consol_supcon_temperature', type=float, default=0.1,
                            help='SupCon temperature for projection training.')
        parser.add_argument('--consol_top_k', type=int, default=3,
                            help='Top-k soft-attention retrieval over bank slots.')
        parser.add_argument('--consol_retrieval_temperature', type=float, default=0.1,
                            help='Softmax temperature over top-k slot similarities.')
        # Slot-recruitment mechanisms (anti-collapse; all off by default).
        parser.add_argument('--consol_blend_floor', type=float, default=0.0,
                            help='Lower bound on the weight-scaled bank-EMA blend for '
                                 'touched slots; 1.0 reproduces the pre-2026-06-10 '
                                 'full-strength recruitment dynamics (A/B knob).')
        parser.add_argument('--consol_data_init', type=binary_to_boolean_type, default=0,
                            help='Seed bank keys from the first batches of real encoded '
                                 'chunks (+1%% noise) instead of random-normal init.')
        parser.add_argument('--consol_dead_restart_interval', type=int, default=0,
                            help='Every N steps, reseed slots whose usage grew less than '
                                 '--consol_dead_min_hits (VQ-VAE-style dead-code restart). '
                                 '0 disables.')
        parser.add_argument('--consol_dead_min_hits', type=float, default=1.0,
                            help='Minimum accumulated retrieval weight per restart '
                                 'interval for a slot to count as alive.')
        parser.add_argument('--consol_dead_protect_hits', type=float, default=50.0,
                            help='Slots with LIFETIME usage above this are never '
                                 'reseeded by dead restarts (protects earlier '
                                 'tasks\' slots from being recycled).')
        parser.add_argument('--consol_novelty_threshold', type=float, default=-1.0,
                            help='Samples whose top-1 slot similarity is below this '
                                 'claim the least-used slot (key overwritten, values '
                                 'reset). -1 disables; adds one GPU sync per step '
                                 'when enabled.')
        parser.add_argument('--consol_use_batched', type=binary_to_boolean_type, default=1,
                            help='Use the cross-layer StackedConsolidation (default); '
                                 '0 falls back to per-layer modules.')
        parser.add_argument('--consol_seed', type=int, default=-1,
                            help='Bank init seed; -1 = use --seed (or 0 if unset).')
        parser.add_argument('--consol_bf16', type=binary_to_boolean_type, default=0,
                            help='Run the backbone forward under bf16 autocast (CUDA only). '
                                 'Off by default for parity with fp32 baselines.')
        parser.add_argument('--consol_log_every', type=int, default=100,
                            help='Steps between consolidation diagnostic logs '
                                 '(diagnostics force GPU syncs; keep sparse).')
        # Slot-prioritized replay (anti-forgetting; off by default).
        # Per-layer path only for now (--consol_use_batched 0).
        parser.add_argument('--consol_replay_mode', type=str, default='off',
                            choices=['off', 'head', 'weight', 'both'],
                            help='Replay mode. head = synthetic CLS features '
                                 'through the current head (fixes Class-IL); '
                                 'weight = re-drive the qkv write from stored '
                                 '(key,value) [step 5, not yet implemented]; '
                                 'both = both. Requires --consol_use_batched 0.')
        parser.add_argument('--consol_replay_every', type=int, default=1,
                            help='Run a replay step every N observe() batches.')
        parser.add_argument('--consol_replay_m', type=int, default=64,
                            help='Slots drawn per replay step (with repetition).')
        parser.add_argument('--consol_replay_weight', type=float, default=1.0,
                            help='Weight on the replay cross-entropy loss.')
        parser.add_argument('--consol_replay_anchor_weight', type=float, default=1.0,
                            help='Weight on the decoder anchoring MSE.')
        parser.add_argument('--consol_replay_min_usage', type=float, default=1.0,
                            help='Lifetime-usage floor below which a slot is '
                                 'ineligible for replay.')
        parser.add_argument('--consol_replay_w_single', type=float, default=1.0,
                            help='Weight on sqrt(current-task hits) in priority.')
        parser.add_argument('--consol_replay_w_cross', type=float, default=0.3,
                            help='Weight on log1p(prior-task hits) in priority.')
        parser.add_argument('--consol_replay_decode_layer', type=int, default=-1,
                            help='Hooked-layer index whose bank feeds the '
                                 'decoder; -1 = last hooked layer.')
        parser.add_argument('--consol_replay_update_backbone',
                            type=binary_to_boolean_type, default=0,
                            help='Reserved (weight-mode/future): head-mode '
                                 'never routes gradient through the backbone, '
                                 'so the backbone is frozen during head replay '
                                 'regardless of this flag.')
        parser.add_argument('--consol_replay_weight_alpha_floor', type=float,
                            default=5e-4,
                            help='Weight-mode consolidation strength floor '
                                 '(step 5, not yet implemented).')
        parser.add_argument('--consol_decoder_hidden', type=int, default=384,
                            help='Replay decoder MLP hidden width.')
        parser.add_argument('--consol_decoder_noise', type=float, default=0.0,
                            help='Std of Gaussian noise injected in the decoder '
                                 'hidden activation during replay generation.')
        parser.add_argument('--consol_decoder_summary', type=str, default='mean',
                            choices=['mean', 'cls', 'cls+mean'],
                            help="Decoder-input summary of a slot's stored "
                                 'values. mean = per-head token-mean (dilutes '
                                 'CLS); cls = the CLS token slice (chunk 0, '
                                 'token 0) — already stored, just selected, '
                                 'and most predictive of the CLS-feature '
                                 'anchor target; cls+mean = both concatenated.')
        parser.add_argument('--consol_ncm_canary', type=binary_to_boolean_type, default=0,
                            help='Drift canary: accumulate per-class means of '
                                 'real CLS features online, and at each task '
                                 'boundary log nearest-class-mean Class-IL/'
                                 'Task-IL accuracy over the test sets. If the '
                                 'head accuracy rises but NCM does not track, '
                                 'replay is drifting onto synthetic features '
                                 '(back off --consol_replay_m / _weight).')
        parser.add_argument('--consol_ncm_oracle', type=binary_to_boolean_type, default=0,
                            help='Oracle NCM probe: at each task boundary, '
                                 'recompute class means from the CURRENT '
                                 "backbone's features over the test sets and "
                                 'classify by nearest mean. Removes the online '
                                 "canary's stale-mean confound, so old-task "
                                 'accuracy reflects whether the final features '
                                 'still separate old classes (an upper bound: '
                                 'if even this is ~0, the representation is '
                                 'genuinely overwritten). Adds one test-set '
                                 'feature pass per task boundary.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        blocks = list(self.net.blocks)
        assert len(blocks) > 0 and hasattr(blocks[0].attn, 'qkv'), \
            "consolidation_vit needs a ViT backbone with fused-qkv attention " \
            "(e.g. --backbone vit-s-16 or vit)."
        self._qkv_layers = [blk.attn.qkv for blk in blocks]
        self._embed_dim = self._qkv_layers[0].in_features
        self._num_heads = blocks[0].attn.num_heads
        assert all(q.out_features == 3 * self._embed_dim for q in self._qkv_layers)

        # Consolidation state is bound lazily (see module docstring).
        self.consol = None            # StackedConsolidation when batched
        self.consol_modules = []      # per-layer ConsolidationModules otherwise
        self.projection_optimizer = None

        # Slot-prioritized replay state (set in _setup_consolidation when
        # --consol_replay_mode != off). _decode_layer indexes the per-layer
        # module that holds the decoder; replay_head_optimizer updates the
        # classifier head on the replay loss (separate from the main
        # optimizer Mammoth resets each task).
        self._replay_on = False
        self._replay_modes = ()
        self._head_replay = False
        self._decode_layer = -1
        self._replay_module = None
        self.replay_head_optimizer = None

        # NCM drift canary: online running sums of real CLS features per
        # class (mean = sum / count), allocated in _setup_consolidation
        # when --consol_ncm_canary is set.
        self._ncm_canary = False
        self._ncm_oracle = False
        self.ncm_sum = None
        self.ncm_count = None

        # Hook plumbing: hooks only cache while observe() runs a forward,
        # so evaluation passes cost nothing. Custom eval scripts (probes,
        # TTC) can flip this via set_caching().
        self._caching = False
        self._labels = None
        self._observe_step = 0
        for i, qkv in enumerate(self._qkv_layers):
            qkv.register_forward_hook(self._make_qkv_hook(i))

        # Error accounting (mirrors ConsolidationCL): consolidate() errors
        # are caught so one bad step doesn't kill a run, but they are
        # counted and surfaced. A silent swallow once hid that an entire
        # eval run was a no-op.
        self.consolidation_calls = 0
        self.consolidation_errors = 0
        self.first_consolidation_error = None
        self._last_error_warn_at = 0

    # ------------------------------------------------------------------
    # Hooks and lazy construction
    # ------------------------------------------------------------------

    def _make_qkv_hook(self, layer: int):
        def hook(module, inputs, output):
            if not self._caching:
                return
            D = self._embed_dim
            X = inputs[0]
            Y_q, Y_k, Y_v = output[..., :D], output[..., D:2 * D], output[..., 2 * D:]
            if self.consol is not None:
                self.consol.cache(layer, X, Y_q, Y_k, Y_v, labels=self._labels)
            else:
                self.consol_modules[layer].cache(X, Y_q, Y_k, Y_v, labels=self._labels)
        return hook

    def set_caching(self, enabled: bool):
        """Enable/disable cache-filling on forward (for external eval scripts)."""
        self._caching = enabled

    def _fused_weight_slices(self):
        """Q/K/V row-slice views into each fused qkv weight's storage."""
        D = self._embed_dim
        ws = [m.weight.data for m in self._qkv_layers]
        return ([w[:D] for w in ws], [w[D:2 * D] for w in ws], [w[2 * D:] for w in ws])

    def _setup_consolidation(self, steps_per_epoch: int):
        args = self.args
        replay_on = args.consol_replay_mode != 'off'
        # head-mode needs the decoder + class histograms; weight-mode needs
        # neither (it re-ascends the stashed consolidation delta).
        head_on = args.consol_replay_mode in ('head', 'both')
        n_patches = self.net.patch_embed.num_patches
        seq_length = n_patches + getattr(self.net, 'num_prefix_tokens', 1)
        # Optional padding to a composite length so chunk counts that the
        # (often prime) backbone sequence length forbids become reachable.
        if args.consol_max_seq_length > 0:
            max_seq_length = args.consol_max_seq_length
            if max_seq_length < seq_length:
                raise ValueError(
                    f"--consol_max_seq_length ({max_seq_length}) must be >= the backbone "
                    f"sequence length ({seq_length})")
            if max_seq_length > seq_length and args.consol_use_batched:
                # The batched cross-layer kernel asserts N == max_seq_length and
                # carries no pad mask; padding only works on the per-layer path.
                raise ValueError(
                    f"--consol_max_seq_length ({max_seq_length}) pads beyond the backbone "
                    f"sequence length ({seq_length}), which the batched path does not "
                    f"support. Re-run with --consol_use_batched 0.")
        else:
            max_seq_length = seq_length
        chunk_length = args.consol_chunk_length if args.consol_chunk_length > 0 else max_seq_length
        warmup_steps = args.consol_warmup_steps if args.consol_warmup_steps >= 0 \
            else int(args.consol_warmup_epochs * steps_per_epoch)
        projection_dim = args.consol_projection_dim if args.consol_projection_dim > 0 else None
        seed = args.consol_seed if args.consol_seed >= 0 \
            else (args.seed if args.seed is not None else 0)
        d_head = self._embed_dim // self._num_heads
        weights_q, weights_k, weights_v = self._fused_weight_slices()

        # Decode layer that holds the replay decoder / class histograms
        # (last hooked layer by default). Used by both backends.
        n_layers = len(self._qkv_layers)
        decode_idx = (args.consol_replay_decode_layer
                      if args.consol_replay_decode_layer >= 0
                      else n_layers + args.consol_replay_decode_layer)

        common = dict(
            input_dim=self._embed_dim,
            num_heads=self._num_heads,
            d_head=d_head,
            max_seq_length=max_seq_length,
            chunk_length=chunk_length,
            pad_mode=args.consol_pad_mode,
            n_slots=args.consol_n_slots,
            alpha=args.consol_alpha,
            beta_key=args.consol_beta_key,
            beta_value=args.consol_beta_value,
            warmup_steps=warmup_steps,
            normalize_delta=args.consol_normalize_delta,
            projection_dim=projection_dim,
            projection_hidden=args.consol_projection_hidden,
            projection_train_steps=args.consol_projection_train_steps,
            projection_train_always=args.consol_projection_train_always,
            supcon_temperature=args.consol_supcon_temperature,
            top_k=args.consol_top_k,
            retrieval_temperature=args.consol_retrieval_temperature,
            blend_floor=args.consol_blend_floor,
            data_init=args.consol_data_init,
            dead_restart_interval=args.consol_dead_restart_interval,
            dead_min_hits=args.consol_dead_min_hits,
            dead_protect_hits=args.consol_dead_protect_hits,
            novelty_threshold=args.consol_novelty_threshold,
            cache_projections=True,
        )

        if args.consol_use_batched:
            # Replay bookkeeping lives on the decode layer's slice only
            # (n_classes=0 => byte-identical to non-replay runs).
            self.consol = StackedConsolidation(
                n_layers=n_layers,
                weights_q=weights_q, weights_k=weights_k, weights_v=weights_v,
                seed=seed,
                n_classes=(self.num_classes if head_on else 0),
                decode_layer=decode_idx,
                **common,
            ).to(self.device)
            proj_params = self.consol.projection_params
        else:
            # Replay bookkeeping (slot_class_hist / task_hit_count) is only
            # needed on the decode layer's bank; off (n_classes=0) elsewhere
            # so other layers stay byte-identical to non-replay runs.
            self.consol_modules = torch.nn.ModuleList()
            for i in range(len(self._qkv_layers)):
                n_classes_i = self.num_classes if (head_on and i == decode_idx) else 0
                cm = ConsolidationModule(
                    weight_q=weights_q[i], weight_k=weights_k[i], weight_v=weights_v[i],
                    seed=seed + i, n_classes=n_classes_i, **common,
                ).to(self.device)
                self.consol_modules.append(cm)
            proj_params = [p for cm in self.consol_modules
                           if cm.projection is not None
                           for p in cm.projection.parameters()]

        # Separate optimizer for the learned key projections; persists
        # across tasks (the main optimizer is reset by Mammoth each task).
        if proj_params:
            self.projection_optimizer = torch.optim.AdamW(
                proj_params, lr=args.consol_projection_lr, weight_decay=0.0)

        # Slot-prioritized replay: build the decoder on the decode layer
        # and a dedicated AdamW for the classifier head (the main optimizer
        # is reset each task; the head's replay updates must persist).
        if replay_on:
            self._replay_on = True
            self._decode_layer = decode_idx
            self._replay_modes = {
                'head': ('head',), 'weight': ('weight',),
                'both': ('head', 'weight'),
            }[args.consol_replay_mode]
            self._head_replay = 'head' in self._replay_modes
            # Unified replay handle: the StackedConsolidation (batched) or
            # the decode-layer ConsolidationModule (per-layer) both expose
            # the same setup_decoder / replay_step / reset_task_hits API.
            self._replay_module = (self.consol if args.consol_use_batched
                                   else self.consol_modules[decode_idx])
            if self._head_replay:
                self._replay_module.setup_decoder(
                    feature_dim=self._embed_dim,
                    hidden=args.consol_decoder_hidden,
                    noise=args.consol_decoder_noise,
                    lr=args.consol_projection_lr,
                    summary_mode=args.consol_decoder_summary,
                )
                self.replay_head_optimizer = torch.optim.AdamW(
                    self.net.head.parameters(), lr=args.lr, weight_decay=0.0)
            logging.info(
                f"consolidation_vit: replay mode={args.consol_replay_mode} "
                f"decode_layer={decode_idx} m={args.consol_replay_m} "
                f"every={args.consol_replay_every} "
                f"w_single={args.consol_replay_w_single} "
                f"w_cross={args.consol_replay_w_cross} "
                f"min_usage={args.consol_replay_min_usage} "
                f"decoder_noise={args.consol_decoder_noise} "
                f"decoder_summary={args.consol_decoder_summary} "
                f"alpha_floor={args.consol_replay_weight_alpha_floor}")

        if args.consol_ncm_canary:
            self._ncm_canary = True
            self.ncm_sum = torch.zeros(
                self.num_classes, self._embed_dim, device=self.device)
            self.ncm_count = torch.zeros(self.num_classes, device=self.device)
            logging.info("consolidation_vit: NCM drift canary enabled.")
        if args.consol_ncm_oracle:
            self._ncm_oracle = True
            logging.info("consolidation_vit: NCM oracle probe enabled.")

        logging.info(
            f"consolidation_vit: {len(self._qkv_layers)} fused-qkv layers hooked "
            f"({'batched cross-layer' if args.consol_use_batched else 'per-layer'} mode); "
            f"D={self._embed_dim} H={self._num_heads} N={max_seq_length} L={chunk_length} "
            f"slots={args.consol_n_slots} alpha={args.consol_alpha} "
            f"warmup={warmup_steps} steps top_k={args.consol_top_k} "
            f"projection_dim={projection_dim} | recruitment: "
            f"blend_floor={args.consol_blend_floor} "
            f"data_init={args.consol_data_init} "
            f"dead_restart={args.consol_dead_restart_interval}"
            f"@{args.consol_dead_min_hits} "
            f"novelty={args.consol_novelty_threshold}")

    def begin_task(self, dataset):
        if self.consol is None and not self.consol_modules:
            self._setup_consolidation(steps_per_epoch=len(dataset.train_loader))
        # Reset the per-task hit counter so P_j = prior-task hits is correct
        # from the first batch of the new task. Only head-mode tracks it.
        if self._head_replay:
            self._replay_module.reset_task_hits()

    def end_task(self, dataset):
        summary = self.error_summary()
        if summary['consolidation_errors'] > 0:
            logging.warning(f"consolidation_vit task {self.current_task}: {summary}")
        else:
            logging.info(f"consolidation_vit task {self.current_task}: "
                         f"{summary['consolidation_calls']} consolidation calls, 0 errors")
        if self._ncm_canary:
            self._run_ncm_canary(dataset)
        if self._ncm_oracle:
            self._run_ncm_oracle(dataset)

    # ------------------------------------------------------------------
    # NCM drift canary
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_ncm_canary(self, dataset):
        """Classify real test CLS features by nearest class-mean and log
        Class-IL / Task-IL NCM accuracy over all seen tasks.

        The means come from real training features (online), so NCM
        accuracy measures backbone representation quality / drift,
        independent of the classifier head. Comparing its trajectory to
        the head accuracy is the canary: head up but NCM flat => the head
        is fitting synthetic replay features rather than the real manifold.
        """
        valid = self.ncm_count > 0
        if int(valid.sum().item()) == 0:
            return
        means = torch.zeros_like(self.ncm_sum)
        means[valid] = self.ncm_sum[valid] / self.ncm_count[valid].unsqueeze(-1)

        was_training = self.net.training
        self.net.eval()
        cil_correct = til_correct = total = 0
        per_task_cil = []
        try:
            for t, loader in enumerate(dataset.test_loaders):
                s, e = self.get_offsets(t)
                tc = te = 0
                for data in loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    feats = self.net(inputs, returnt='features').float()
                    dist = torch.cdist(feats, means)          # (B, C)
                    dist[:, ~valid] = float('inf')
                    cil_pred = dist.argmin(dim=1)
                    # Task-IL: restrict to this task's class window.
                    dist_til = torch.full_like(dist, float('inf'))
                    dist_til[:, s:e] = dist[:, s:e]
                    til_pred = dist_til.argmin(dim=1)
                    cil_correct += int((cil_pred == labels).sum().item())
                    til_correct += int((til_pred == labels).sum().item())
                    tc += int((cil_pred == labels).sum().item())
                    te += int(labels.numel())
                    total += int(labels.numel())
                per_task_cil.append(100.0 * tc / max(te, 1))
        finally:
            if was_training:
                self.net.train()

        cil = 100.0 * cil_correct / max(total, 1)
        til = 100.0 * til_correct / max(total, 1)
        raw = ', '.join(f'{a:.2f}' for a in per_task_cil)
        logging.info(
            f"NCM canary task {self.current_task}: "
            f"[Class-IL] {cil:.2f}%  [Task-IL] {til:.2f}%  "
            f"| per-task CIL [{raw}]")

    @torch.no_grad()
    def _run_ncm_oracle(self, dataset):
        """Oracle NCM: build class means from the CURRENT backbone's features
        over the test sets, then classify those features by nearest mean.

        Unlike the online canary (means captured during each class's task,
        hence stale once the backbone drifts), this recomputes means from
        the current representation — so old-task accuracy reflects whether
        the final features still separate old classes, with no stale-mean
        confound. It is an upper bound (means built on the same test data,
        and the model is told each sample's task for Task-IL): if even this
        is ~0 on old tasks, the representation is genuinely overwritten and
        no head-side intervention can recover those classes.
        """
        was_training = self.net.training
        self.net.eval()
        feats_all, labels_all, tasks_all = [], [], []
        try:
            for t, loader in enumerate(dataset.test_loaders):
                for data in loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    f = self.net(inputs, returnt='features').float()
                    feats_all.append(f)
                    labels_all.append(labels)
                    tasks_all.append(torch.full_like(labels, t))
        finally:
            if was_training:
                self.net.train()
        if not feats_all:
            return

        feats = torch.cat(feats_all)                          # (N, D)
        labels = torch.cat(labels_all)                        # (N,)
        tasks = torch.cat(tasks_all)                          # (N,)
        n_tasks = len(dataset.test_loaders)

        # Class means from the current features.
        sums = torch.zeros(self.num_classes, feats.shape[1], device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
        valid = counts > 0
        means = torch.zeros_like(sums)
        means[valid] = sums[valid] / counts[valid].unsqueeze(-1)

        dist = torch.cdist(feats, means)                      # (N, C)
        dist[:, ~valid] = float('inf')
        cil_hit = dist.argmin(dim=1) == labels                # (N,)

        # Task-IL: restrict each sample to its own task's class window.
        til_dist = torch.full_like(dist, float('inf'))
        for t in range(n_tasks):
            s, e = self.get_offsets(t)
            sel = tasks == t
            til_dist[sel, s:e] = dist[sel, s:e]
        til_hit = til_dist.argmin(dim=1) == labels

        N = feats.shape[0]
        cil = 100.0 * int(cil_hit.sum().item()) / max(N, 1)
        til = 100.0 * int(til_hit.sum().item()) / max(N, 1)
        per_task = []
        for t in range(n_tasks):
            sel = tasks == t
            n = int(sel.sum().item())
            per_task.append(100.0 * int(cil_hit[sel].sum().item()) / max(n, 1))
        raw = ', '.join(f'{a:.2f}' for a in per_task)
        logging.info(
            f"NCM ORACLE task {self.current_task}: "
            f"[Class-IL] {cil:.2f}%  [Task-IL] {til:.2f}%  "
            f"| per-task CIL [{raw}]")

    # ------------------------------------------------------------------
    # Consolidation step plumbing (mirrors ConsolidationCL)
    # ------------------------------------------------------------------

    def _record_error(self, where: str, e: Exception):
        self.consolidation_errors += 1
        if self.first_consolidation_error is None:
            self.first_consolidation_error = f"{where}: {type(e).__name__}: {e}"
        if (self.consolidation_errors == 1
                or self.consolidation_errors - self._last_error_warn_at >= 1000):
            self._last_error_warn_at = self.consolidation_errors
            logging.error(
                f"consolidation_vit: consolidate() failure #{self.consolidation_errors} "
                f"(of {self.consolidation_calls} calls) at {where}: "
                f"{type(e).__name__}: {e}")

    def error_summary(self):
        return {
            'consolidation_calls': self.consolidation_calls,
            'consolidation_errors': self.consolidation_errors,
            'first_consolidation_error': self.first_consolidation_error,
        }

    def _train_projection_step(self, want_diagnostics: bool):
        if self.projection_optimizer is None:
            return {}
        if self.consol is not None:
            total, per_layer = self.consol.compute_supcon_loss()
            if total is None:
                return {}
        else:
            losses = [l for l in (cm.compute_supcon_loss() for cm in self.consol_modules)
                      if l is not None]
            if not losses:
                return {}
            total = sum(losses) / len(losses)
        self.projection_optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.projection_optimizer.step()
        if not want_diagnostics:
            return {}
        return {'supcon_mean': float(total.item())}

    def _consolidate(self, lr_now: float, want_diagnostics: bool):
        self.consolidation_calls += 1
        if self.consol is not None:
            try:
                return self.consol.consolidate_all(
                    lr_now=lr_now, lr_anchor=self.args.lr,
                    compute_separation_metrics=want_diagnostics,
                    want_diagnostics=want_diagnostics)
            except Exception as e:
                self._record_error('stacked', e)
                return []
        diags = []
        for i, cm in enumerate(self.consol_modules):
            try:
                d = cm.consolidate(
                    lr_now=lr_now, lr_anchor=self.args.lr,
                    compute_separation_metrics=want_diagnostics,
                    want_diagnostics=want_diagnostics)
                if want_diagnostics:
                    diags.append(d)
            except Exception as e:
                self._record_error(f'layer{i}', e)
        return diags

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        assert self.consol is not None or self.consol_modules, \
            "begin_task must run before observe (lazy consolidation setup)"

        self._labels = labels
        self.opt.zero_grad()
        self._caching = True
        # When replaying we need the real pre-head CLS feature for the
        # decoder anchoring term. returnt='both' returns it from the same
        # forward at no extra cost (the head already consumes it).
        real_feats = None
        want_feats = self._head_replay or self._ncm_canary
        try:
            if self.args.consol_bf16 and inputs.is_cuda:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if want_feats:
                        outputs, feats = self.net(inputs, returnt='both')
                    else:
                        outputs = self.net(inputs)
                    loss = self.loss(outputs, labels)
            else:
                if want_feats:
                    outputs, feats = self.net(inputs, returnt='both')
                else:
                    outputs = self.net(inputs)
                loss = self.loss(outputs, labels)
            if want_feats:
                # Transient: detached and float (decoder is fp32); never stored.
                real_feats = feats.detach().float()
        finally:
            self._caching = False

        # NCM canary: accumulate online per-class running sums of the real
        # CLS features (no host sync; cheap index_add).
        if self._ncm_canary and real_feats is not None:
            with torch.no_grad():
                idx = labels.view(-1).to(torch.long)
                self.ncm_sum.index_add_(0, idx, real_feats)
                self.ncm_count.index_add_(
                    0, idx, torch.ones_like(idx, dtype=self.ncm_count.dtype))

        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        self._observe_step += 1
        want_diag = (self._observe_step % self.args.consol_log_every) == 0

        # SupCon step BEFORE consolidate(): it reads the same cached
        # X/labels, and consolidate() clears the cache when done.
        proj_diag = self._train_projection_step(want_diag)
        lr_now = self.opt.param_groups[0]['lr']
        diags = self._consolidate(lr_now, want_diag)

        if want_diag and diags:
            d0 = diags[0]
            msg = (f"consol step={self._observe_step} "
                   f"s={d0['s_mean']:+.3f} vcos_q={d0['vcos_q']:+.3f} "
                   f"dq={d0['delta_norm_q']:.2e} slots={d0['unique_slots_hit']} "
                   f"topk_slots={d0.get('unique_topk_slots', '-')} "
                   f"a_eff={d0['alpha_eff']:.2e} warmup={d0['in_warmup']}")
            if len(diags) > 6:
                d6 = diags[6]
                msg += (f" | L6 s={d6['s_mean']:+.3f} vcos_q={d6['vcos_q']:+.3f} "
                        f"slots={d6['unique_slots_hit']} "
                        f"topk_slots={d6.get('unique_topk_slots', '-')}")
            if proj_diag:
                msg += f" | supcon={proj_diag['supcon_mean']:.3f}"
            if self.consolidation_errors:
                msg += f" | ERRORS={self.consolidation_errors}"
            logging.info(msg)

            # Top-k retrieval diagnostics (softmax diversity): per-rank raw
            # sims and softmax weights, and the weight entropy (0 = argmax-
            # like, log(k) = uniform). The headline for slot collapse: are
            # ranks 2-3 getting any meaningful weight/update mass?
            if d0.get('top_k', 1) > 1:
                def _topk_line(label, d):
                    sims = ', '.join(f'{x:+.3f}' for x in d.get('top_sims_mean_per_rank', []))
                    ws = ', '.join(f'{x:.3f}' for x in d.get('w_top_mean_per_rank', []))
                    return (f"{label} sims=[{sims}] w=[{ws}] "
                            f"H={d.get('w_top_entropy_mean', 0.0):.3f}"
                            f"(max {math.log(d['top_k']):.3f})")
                tk = f"  topk: {_topk_line('L0', d0)}"
                if len(diags) > 6:
                    tk += f" | {_topk_line('L6', diags[6])}"
                logging.info(tk)

            # Input-separation diagnostics (computed when labels were cached):
            # key_cos_margin = within-class minus across-class key cosine
            # (positive = key space separates classes); sep_mcc = slot-
            # agreement vs class-agreement as a binary task.
            if 'key_cos_margin' in d0:
                sep = (f"  sep: L0 margin={d0['key_cos_margin']:+.4f} "
                       f"mcc={d0.get('sep_mcc', float('nan')):+.3f}")
                if len(diags) > 6 and 'key_cos_margin' in diags[6]:
                    sep += (f" | L6 margin={diags[6]['key_cos_margin']:+.4f} "
                            f"mcc={diags[6].get('sep_mcc', float('nan')):+.3f}")
                logging.info(sep)

        # ----- Slot-prioritized replay -----
        # Runs after consolidate() (which refreshes the decode layer's
        # _last_j_star, the slot histograms, and each layer's stashed
        # consolidation delta for this batch).
        if self._replay_on and (self._observe_step % self.args.consol_replay_every == 0):
            # head-mode: CE on prioritized synthetic features (from task 2)
            # + decoder anchoring (from task 1). _replay_module is the
            # StackedConsolidation (batched) or the decode-layer module.
            if 'head' in self._replay_modes:
                dm = self._replay_module
                rdiag = dm.replay_step(
                    head=self.net.head, loss_fn=self.loss,
                    m=self.args.consol_replay_m,
                    replay_weight=self.args.consol_replay_weight,
                    anchor_weight=self.args.consol_replay_anchor_weight,
                    w_single=self.args.consol_replay_w_single,
                    w_cross=self.args.consol_replay_w_cross,
                    min_usage=self.args.consol_replay_min_usage,
                    enable_ce=(self.current_task >= 1),
                    anchor_j_star=dm._last_j_star, anchor_feats=real_feats,
                    head_optimizer=self.replay_head_optimizer,
                    want_diagnostics=want_diag,
                )
                if want_diag and rdiag:
                    logging.info("consol replay " + " ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in rdiag.items()))
            # weight-mode: re-ascend the consolidation delta at the floored,
            # schedule-independent alpha (from task 2). Batched applies all
            # layers in one call; per-layer loops its modules.
            if 'weight' in self._replay_modes and self.current_task >= 1:
                floor = self.args.consol_replay_weight_alpha_floor
                if self.consol is not None:
                    applied = self.consol.weight_replay(floor)
                    total = self.consol.L
                else:
                    applied = sum(cm.weight_replay(floor) for cm in self.consol_modules)
                    total = len(self.consol_modules)
                if want_diag:
                    logging.info(f"consol replay weight: re-ascended "
                                 f"{applied}/{total} layers "
                                 f"@ alpha_floor={floor:.2e}")

        return loss.item()
