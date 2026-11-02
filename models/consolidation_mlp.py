"""
Sparse, match-gated hippocampal consolidation on a standard MLP, as a
Mammoth model. The vector-space reduction of `consolidation_vit`: standard
neurons, no tokens / heads / chunking, one sparse hippocampal bank per
consolidatable Linear layer (instead of 64 discrete slots).

Substrate: the `mnistmlp` backbone (`backbone/MNISTMLP.py`, BaseMNISTMLP):

    fc1 (784->100) -> ReLU -> fc2 (100->100) -> ReLU -> classifier (100->C)

`fc1` and `fc2` are the two consolidatable layers. A forward hook on each
caches that layer's input a_{l-1} and pre-activation output z_l during the
observe() forward; the post-ReLU activation a_l = ReLU(z_l) is what the
consolidation step (step 2+) will associate with the sparse bank.

This file is built incrementally (see HANDOFF_sparse_consolidation_mlp.md
§4), each step flag-gated and default-off, defaults byte-identical to stock
`sgd`:

  * STEP 1 (this commit): scaffold. Hook plumbing + a no-op consolidate
    stub. With `--consol_mlp 0` (default) observe() is the literal `sgd`
    path. With `--consol_mlp 1` the hooks fire and cache (detached, so
    numerically inert) but no weight update is applied yet — so BOTH modes
    reproduce stock `sgd` byte-for-byte. verify_consolidation_mlp.py checks
    this. The sparse bank, the cosine match, the consolidation write, and
    the cross-layer gate land in steps 2-4.

NOTE on the hook gate: hooks only cache while `_caching` is True, which is
set only around observe()'s forward. Evaluation passes therefore cost
nothing, and when consolidation is disabled `_caching` never flips so the
hooks are wholly inert.
"""

import logging

import torch

from models.utils.continual_model import ContinualModel
from models.consolidation_mlp_utils import SparseLayerBank, TaskRouter
from utils.args import ArgumentParser, binary_to_boolean_type


class ConsolidationMlp(ContinualModel):
    """Sparse hippocampal consolidation of MLP layer weights."""

    NAME = 'consolidation_mlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Defaults matching the seq-mnist `sgd` protocol from
        # clser_best_args.py (apples-to-apples with the CLS-ER table):
        # SGD, lr 0.03, batch_size 10, 1 epoch/task.
        parser.set_defaults(optimizer='sgd', lr=0.03)

        # Master switch. Off => observe() is byte-identical to stock `sgd`.
        parser.add_argument('--consol_mlp', type=binary_to_boolean_type, default=0,
                            help='Enable sparse hippocampal consolidation on the '
                                 'MLP. Off (default) reproduces stock `sgd`.')
        # --- Sparse bank (step 2) ---
        parser.add_argument('--consol_sparsity', type=str, default='kwta',
                            choices=['kwta', 'l1'],
                            help='Sparsity mechanism for the layer code: kwta '
                                 '(hard top-N%%, DG-like) or l1 (soft ReLU code, '
                                 'L1-encouraged).')
        parser.add_argument('--consol_sparsity_pct', type=float, default=0.05,
                            help='k-WTA active fraction: k = round(pct * proj_dim).')
        parser.add_argument('--consol_proj_dim', type=int, default=2000,
                            help='Sparse projection dimension (high-dim DG space).')
        parser.add_argument('--consol_proj_learned', type=binary_to_boolean_type,
                            default=0,
                            help='Learn the projection (default: fixed-random, '
                                 'most DG-like and does not drift).')
        parser.add_argument('--consol_beta_bank', type=float, default=0.99,
                            help='EMA momentum for the superposed bank vector.')
        # --- Consolidation write (step 3) ---
        parser.add_argument('--consol_alpha', type=float, default=1e-3,
                            help='Consolidation step size (Frobenius step per '
                                 'layer when normalize_delta).')
        parser.add_argument('--consol_normalize_delta', type=binary_to_boolean_type,
                            default=1,
                            help='Unit-normalize each per-W delta before alpha '
                                 '(so a step moves W by exactly alpha).')
        parser.add_argument('--consol_warmup_epochs', type=float, default=1.0,
                            help='Bank warmup in epochs before any weight write '
                                 '(converted to steps at task 0).')
        parser.add_argument('--consol_warmup_steps', type=int, default=-1,
                            help='Bank warmup in steps; overrides '
                                 '--consol_warmup_epochs if >= 0.')
        # --- Cross-layer gate (step 4) ---
        parser.add_argument('--consol_gate_prev_layer', type=binary_to_boolean_type,
                            default=1,
                            help='Scale layer l\'s consolidation per-sample by the '
                                 'previous layer\'s match s_{l-1} (layer 0 gate=1). '
                                 '0 = ungated (every layer uses s_0=1).')
        # --- Diagnostics (step 5; read-only, run at task boundaries) ---
        parser.add_argument('--consol_ncm_oracle', type=binary_to_boolean_type,
                            default=0,
                            help='Oracle NCM probe on fc2 features: at each task '
                                 'boundary recompute class means from the CURRENT '
                                 'backbone and classify nearest-mean (Class-IL/'
                                 'Task-IL upper bound, head-independent). The MLP '
                                 'analog of the ViT oracle — reveals whether old '
                                 'classes still separate in the representation.')
        parser.add_argument('--consol_probe', type=binary_to_boolean_type, default=0,
                            help='Bank probe: at each task boundary log per-layer '
                                 'code sparsity (active fraction), pairwise '
                                 'crosstalk (mean|cos|), and mean match s_l on a '
                                 'real test batch. Also logs routing accuracy when '
                                 'routing is active.')
        # --- Test-time task routing (per-task buffers; readout fix) ---
        parser.add_argument('--consol_mode', type=str, default='both',
                            choices=['inweight', 'routing', 'both'],
                            help='inweight = the gated in-weight ascent (steps '
                                 '1-5); routing = per-task buffers used only at '
                                 'test time to infer the task and bias the logits '
                                 'to its class window (backbone trains by plain '
                                 'SGD); both = run both (default).')
        parser.add_argument('--consol_route_proj', type=str, default='kwta_random',
                            choices=['kwta_random', 'l1', 'supcon'],
                            help='Per-task routing projection: kwta_random '
                                 '(fixed-random + k-WTA, no training); l1 (learned, '
                                 'L1-sparse code); supcon (learned, supervised '
                                 'contrastive on the task classes).')
        parser.add_argument('--consol_route_dim', type=int, default=2000,
                            help='Routing projection dimension.')
        parser.add_argument('--consol_route_sparsity_pct', type=float, default=0.05,
                            help='Routing k-WTA active fraction.')
        parser.add_argument('--consol_route_beta', type=float, default=0.99,
                            help='EMA momentum for per-task routing buffers.')
        parser.add_argument('--consol_route_lr', type=float, default=1e-3,
                            help='LR for learned routing projections (l1/supcon).')
        parser.add_argument('--consol_route_l1_weight', type=float, default=1e-2,
                            help='L1 penalty weight for the l1 routing projection.')
        parser.add_argument('--consol_route_supcon_temp', type=float, default=0.1,
                            help='SupCon temperature for the supcon routing projection.')
        parser.add_argument('--consol_route_bias', type=float, default=1e4,
                            help='Additive logit bias on the predicted task window '
                                 'at test time (large = hard routing for Class-IL; '
                                 'uniform within a window, so Task-IL is unchanged).')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # The backbone must be the two-hidden-layer MNIST MLP: fc1, fc2 are
        # the consolidatable Linear layers, classifier is the readout head.
        for attr in ('fc1', 'fc2', 'classifier'):
            assert hasattr(self.net, attr), (
                f"consolidation_mlp needs an mnistmlp backbone with a `{attr}` "
                f"Linear (run with --backbone mnistmlp).")
        self._layers = [self.net.fc1, self.net.fc2]

        self._enabled = bool(args.consol_mlp)
        # Sub-behavior: in-weight ascent (steps 1-5), test-time routing, or both.
        self._mode = args.consol_mode if self._enabled else 'inweight'
        self._inweight = self._enabled and self._mode in ('inweight', 'both')
        self._routing = self._enabled and self._mode in ('routing', 'both')
        self.router = None           # TaskRouter when routing is active

        # Consolidation state is bound lazily in begin_task of the first
        # task (mirrors consolidation_vit: .to() re-allocates parameter
        # storage, so weight views must be taken after the backbone has
        # moved to its device).
        self._consol_ready = False
        self.banks = None            # ModuleList[SparseLayerBank], one per layer
        # Most recent per-layer mean match s_l (Python floats), for
        # diagnostics now and the cross-layer gate in step 4.
        self._last_s = [None] * len(self._layers)

        # Hook plumbing: hooks only cache while observe() runs a forward.
        # Per-layer caches of (input a_{l-1}, pre-activation output z_l).
        self._caching = False
        self._labels = None
        self._cache = [None] * len(self._layers)
        self._observe_step = 0
        for i, layer in enumerate(self._layers):
            layer.register_forward_hook(self._make_hook(i))

        if self._enabled:
            logging.info("consolidation_mlp: enabled (step-1 scaffold: hooks "
                         "cache but no weight update is applied yet).")

    # ------------------------------------------------------------------
    # Hooks and lazy construction
    # ------------------------------------------------------------------

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            if not self._caching:
                return
            # inputs[0]: a_{l-1} (B, in_features); output: z_l = W a_{l-1} + b
            # (B, out_features), pre-ReLU. Detach: caching must never touch
            # the main task autograd graph.
            self._cache[layer] = (inputs[0].detach(), output.detach())
        return hook

    def set_caching(self, enabled: bool):
        """Enable/disable cache-filling on forward (for external eval scripts)."""
        self._caching = enabled

    def _setup_consolidation(self, steps_per_epoch: int):
        """Bind the per-layer sparse banks (one per consolidatable Linear).
        A layer's bank codes its OUTPUT activation a_l = ReLU(z_l), so the
        bank's input dim is the layer's out_features."""
        args = self.args
        seed = args.seed if args.seed is not None else 0
        # Warmup: no weight write until the bank has populated. Converted
        # once at task 0 (steps override epochs if >= 0).
        self._warmup_steps = (args.consol_warmup_steps
                              if args.consol_warmup_steps >= 0
                              else int(args.consol_warmup_epochs * steps_per_epoch))

        # In-weight consolidation banks (one per Linear layer). Only built when
        # the in-weight ascent is active.
        if self._inweight:
            banks = torch.nn.ModuleList()
            for i, layer in enumerate(self._layers):
                banks.append(SparseLayerBank(
                    in_dim=layer.out_features,
                    proj_dim=args.consol_proj_dim,
                    sparsity=args.consol_sparsity,
                    sparsity_pct=args.consol_sparsity_pct,
                    proj_learned=bool(args.consol_proj_learned),
                    beta=args.consol_beta_bank,
                    seed=seed + i,
                ))
            self.banks = banks.to(self.device)

        # Per-task routing buffers (test-time task inference on fc2 features).
        if self._routing:
            self.router = TaskRouter(
                feat_dim=self.net.fc2.out_features,
                proj_dim=args.consol_route_dim,
                mode=args.consol_route_proj,
                sparsity_pct=args.consol_route_sparsity_pct,
                beta=args.consol_route_beta,
                l1_weight=args.consol_route_l1_weight,
                supcon_temp=args.consol_route_supcon_temp,
                lr=args.consol_route_lr,
                seed=seed + 100,
            ).to(self.device)

        self._consol_ready = True
        logging.info(
            f"consolidation_mlp: mode={self._mode} "
            f"(inweight={self._inweight} routing={self._routing}); "
            f"steps_per_epoch={steps_per_epoch}. "
            f"inweight banks: proj_dim={args.consol_proj_dim} "
            f"sparsity={args.consol_sparsity}@{args.consol_sparsity_pct} "
            f"alpha={args.consol_alpha} warmup={self._warmup_steps}. "
            f"routing: proj={args.consol_route_proj} dim={args.consol_route_dim} "
            f"bias={args.consol_route_bias}.")

    def begin_task(self, dataset):
        if self._enabled and not self._consol_ready:
            self._setup_consolidation(steps_per_epoch=len(dataset.train_loader))
        # A fresh per-task routing buffer + projection for the task starting now.
        if self._routing:
            self.router.add_task(self.current_task,
                                 self.get_offsets(self.current_task))

    def end_task(self, dataset):
        if not self._enabled:
            return
        if self.args.consol_ncm_oracle:
            self._run_ncm_oracle(dataset)
        if self.args.consol_probe and self.banks is not None:
            self._run_probe(dataset)
        if self.args.consol_probe and self._routing:
            self._run_routing_diag(dataset)

    # ------------------------------------------------------------------
    # Forward (eval-time routing bias)
    # ------------------------------------------------------------------

    def forward(self, x, *args):
        """At EVAL (net in eval mode) with routing active, infer each sample's
        task from the per-task buffers and add a large uniform bias to that
        task's class-window logits — converting Task-IL separability into
        Class-IL. Uniform within a window => Task-IL argmax unchanged. During
        training (net in train mode) or without routing, plain backbone."""
        if (not self._routing) or self.net.training or self.router is None \
                or self.router.n_tasks == 0:
            return self.net(x)
        out, feats = self.net(x, returnt='full')
        bias, _ = self.router.logit_bias(
            feats, out.shape[1], self.args.consol_route_bias)
        return out + bias

    # ------------------------------------------------------------------
    # Diagnostics (read-only, at task boundaries)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_ncm_oracle(self, dataset):
        """Oracle NCM on fc2 features (MLP analog of consolidation_vit's
        _run_ncm_oracle). Recompute class means from the CURRENT backbone over
        all seen test sets, classify by nearest mean. Old-task accuracy here is
        an upper bound on what any head could recover: if even this is ~0, the
        representation itself was overwritten, not just the readout."""
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
        labels = torch.cat(labels_all)
        tasks = torch.cat(tasks_all)
        n_tasks = len(dataset.test_loaders)

        sums = torch.zeros(self.num_classes, feats.shape[1], device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
        valid = counts > 0
        means = torch.zeros_like(sums)
        means[valid] = sums[valid] / counts[valid].unsqueeze(-1)

        dist = torch.cdist(feats, means)                      # (N, C)
        dist[:, ~valid] = float('inf')
        cil_hit = dist.argmin(dim=1) == labels

        til_dist = torch.full_like(dist, float('inf'))
        for t in range(n_tasks):
            s, e = self.get_offsets(t)
            sel = tasks == t
            til_dist[sel, s:e] = dist[sel, s:e]
        til_hit = til_dist.argmin(dim=1) == labels

        N = feats.shape[0]
        cil = 100.0 * int(cil_hit.sum()) / max(N, 1)
        til = 100.0 * int(til_hit.sum()) / max(N, 1)
        per_task = []
        for t in range(n_tasks):
            sel = tasks == t
            n = int(sel.sum())
            per_task.append(100.0 * int(cil_hit[sel].sum()) / max(n, 1))
        raw = ', '.join(f'{a:.2f}' for a in per_task)
        logging.info(
            f"consol NCM ORACLE task {self.current_task}: "
            f"[Class-IL] {cil:.2f}%  [Task-IL] {til:.2f}%  | per-task CIL [{raw}]")
        return {'class_il': cil, 'task_il': til, 'per_task': per_task}

    @torch.no_grad()
    def _run_probe(self, dataset, max_samples: int = 512):
        """Per-layer bank diagnostics on a real test batch: code sparsity
        (active fraction), pairwise crosstalk (mean|cos| over distinct
        samples — superposition health), and mean match s_l. Uses the forward
        hooks to grab each layer's activation."""
        was_training = self.net.training
        self.net.eval()
        # Pull up to max_samples from the most recent task's test set.
        xs, n = [], 0
        for data in dataset.test_loaders[-1]:
            xs.append(data[0])
            n += data[0].shape[0]
            if n >= max_samples:
                break
        x = torch.cat(xs)[:max_samples].to(self.device)

        self._caching = True
        try:
            self.net(x)
        finally:
            self._caching = False
            if was_training:
                self.net.train()

        msgs, stats = [], []
        for i, bank in enumerate(self.banks):
            cached = self._cache[i]
            if cached is None:
                continue
            _, z_l = cached
            phi = bank.encode(torch.relu(z_l))                # (B, P)
            active = float((phi != 0).float().mean())         # active fraction
            s_l = float(bank.match(phi).mean())
            phin = torch.nn.functional.normalize(phi, dim=-1)
            cos = phin @ phin.t()
            eye = torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
            crosstalk = float(cos[~eye].abs().mean())
            stats.append({'layer': i, 'active': active,
                          'crosstalk': crosstalk, 's': s_l})
            msgs.append(f"L{i}: active={active:.3f} crosstalk={crosstalk:.3f} "
                        f"s={s_l:+.3f}")
        self._clear_cache()
        logging.info(f"consol PROBE task {self.current_task}: " + " | ".join(msgs))
        return stats

    @torch.no_grad()
    def _run_routing_diag(self, dataset):
        """Test-time routing accuracy: over every seen task's test set, predict
        the task from the per-task buffers and compare to the true task. This is
        the ceiling on the Class-IL gain routing can deliver (route correctly ->
        the head's Task-IL accuracy becomes Class-IL accuracy)."""
        was_training = self.net.training
        self.net.eval()
        per_task, correct, total = [], 0, 0
        try:
            for t, loader in enumerate(dataset.test_loaders):
                tc = tt = 0
                for data in loader:
                    inputs = data[0].to(self.device)
                    feats = self.net(inputs, returnt='features').float()
                    pred, _ = self.router.route(feats)
                    tc += int((pred == t).sum())
                    tt += int(inputs.shape[0])
                per_task.append(100.0 * tc / max(tt, 1))
                correct += tc
                total += tt
        finally:
            if was_training:
                self.net.train()
        acc = 100.0 * correct / max(total, 1)
        raw = ', '.join(f'{a:.2f}' for a in per_task)
        logging.info(f"consol ROUTING task {self.current_task}: "
                     f"routing-acc {acc:.2f}%  | per-task [{raw}]")
        return {'routing_acc': acc, 'per_task': per_task}

    # ------------------------------------------------------------------
    # Consolidation step (stub for step 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _consolidate(self):
        """Apply one consolidation step per layer.

        STEP 3-4: for each layer compute the sparse code phi_l of the output
        activation a_l = ReLU(z_l), the cosine match s_l, and EMA-superpose
        phi_l into the bank. THEN, past warmup, ASCEND the match by writing
        W_l += alpha * normalize(G_z^T @ a_{l-1})  in place, where
        G_z = d(sum_b gate_b * s_b)/d z_l.

        STEP 4 cross-layer gate: gate_b = s_{l-1}^{(b)}, the PREVIOUS layer's
        per-sample match computed in this same batch (layer 0 ungated, so
        s_0 = 1). A sample is consolidated deeper only to the extent it was
        recognized at the prior stage. s_{l-1} is cosine in [-1, 1] and is
        reused directly (no recompute). Disabled (--consol_gate_prev_layer 0)
        falls back to the step-3 ungated write.
        """
        in_warmup = self._observe_step <= self._warmup_steps
        alpha = self.args.consol_alpha
        gate_on = bool(self.args.consol_gate_prev_layer)
        s_prev = None                        # previous layer's per-sample s_{l-1}
        for i, (bank, layer) in enumerate(zip(self.banks, self._layers)):
            cached = self._cache[i]
            if cached is None:
                s_prev = None
                continue
            a_prev, z_l = cached             # input a_{l-1}, pre-activation z_l

            # Gate this layer's ascent by the previous layer's match.
            gate = s_prev if (gate_on and i > 0) else None
            G_z, s_l = bank.consolidation_grad(z_l, gate=gate)
            self._last_s[i] = float(s_l.mean())
            s_prev = s_l                     # (B,) detached, reused next layer

            # EMA-superpose this batch's code into the bank.
            bank.update(bank.encode(torch.relu(z_l)))

            if in_warmup or alpha <= 0:
                continue
            # Chain rule: d s / d W_l = G_z^T @ a_{l-1}  (out, in).
            delta_W = G_z.t() @ a_prev
            if self.args.consol_normalize_delta:
                delta_W = delta_W / (delta_W.norm() + 1e-6)
            layer.weight.data.add_(delta_W, alpha=alpha)   # ascent: raise s_l
        self._clear_cache()

    def _clear_cache(self):
        self._cache = [None] * len(self._layers)
        self._labels = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        if not self._enabled:
            # Byte-identical to models/sgd.py.
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()
            return loss.item()

        assert self._consol_ready, \
            "begin_task must run before observe (lazy consolidation setup)"

        self._labels = labels.detach()
        self.opt.zero_grad()
        self._caching = True
        feats = None
        try:
            if self._routing:
                # returnt='full' returns the fc2 representation for the router
                # at no extra cost (the head already consumes it).
                outputs, feats = self.net(inputs, returnt='full')
            else:
                outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
        finally:
            self._caching = False

        loss.backward()
        self.opt.step()

        self._observe_step += 1
        # In-weight ascent (uses the hooked per-layer cache; clears it).
        if self._inweight:
            self._consolidate()
        else:
            self._clear_cache()
        # Routing: train the current task's projection + EMA-update its buffer.
        # Read-only to the backbone (operates on detached features).
        if self._routing:
            self.router.observe(feats.detach(), labels)
        return loss.item()
