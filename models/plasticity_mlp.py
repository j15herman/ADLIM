"""Per-weight metaplasticity on a two-hidden-layer MLP (Mammoth model `plasticity-mlp`).

Backbone: `mnistmlp`, fc1 (784->100) -> ReLU -> fc2 (100->100) -> ReLU -> classifier.
Every weight W_ji of the metaplastic layers (fc1, fc2; also the classifier under
`--plast_layers all`) carries a plasticity p_ji that scales its SGD step,
W <- W - lr * p * g. Each training step:

  1. Signal. Forward hooks cache each layer's input a and pre-activation z. The
     per-weight signal is s_ji = a_i * 1[z_j > 0] (= d a_post_j / d W_ji). Its EMA
     mean M (momentum --plast_beta) gives the importance Omega = |M|.
  2. Spillover. Omega is smoothed along a 1-D dendrite per postsynaptic neuron
     (DendriteGeometry): K(i,k) = exp(-|u_i - u_k| / sigma), where barriers in the
     warped coordinate u carve compartments. Mixed in with fraction --plast_spread,
     layer mean preserved.
  3. Accumulation. A += G * (Omega - mean(Omega)) and S += mean(Omega). G is the
     consolidation-dependent stiffness gain: small for frozen weights (sticky),
     large for plastic ones. The anti-erosion floor keeps a once-important weight's
     A above floor * peak(A).
  4. Plasticity. p = clamp(exp(-lambda * A / S), p_min, p_max): persistently
     important weights freeze (p -> p_min), unimportant ones speed up (p > 1).
  5. Budget. Each neuron's freezing mass sum_i max(1 - p_ji, 0) is capped at
     budget_frac * fan_in (ramped over the run); over-budget rows are softened.
  6. Step. weight.grad *= p; classifier rows of finished tasks get zero grad
     (--plast_head_consolidate past_blocks); cosine LR; optimizer step.
  7. Neuron death. Neurons that stay in the layer's most-plastic tail for
     --plast_death_window steps are re-initialized: incoming weights, importance,
     dendrite, and their outgoing column in the next layer.

Optional extras: one of DER++ / CLS-ER / GEM replay combined with the brake, and
read-only diagnostics logged at task boundaries. Replay forwards run with the hooks
off, so Omega is always measured on the current batch only.

--plast_mlp 0 is plain SGD. The other defaults are the paper configuration, e.g.

    python main.py --model plasticity-mlp --plast_mlp 1 --dataset seq-mnist \
        --backbone mnistmlp --batch_size 10 --n_epochs 1

Options that were tried and dropped are listed in PLASTICITY_MLP_REMOVED.md.
"""

import logging
import math
import os
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from models.plasticity_mlp_utils import DendriteGeometry
from models.gem import store_grad, overwrite_grad, project2cone2
from utils.args import ArgumentParser, binary_to_boolean_type
from utils.buffer import Buffer, fill_buffer

_EPS = 1e-8
# A weight with p below this counts as "frozen" in the probes and turnover dumps.
_FREEZE_THRESH = 0.1
_LAYER_NAMES = ['fc1', 'fc2', 'classifier']


class PlasticityMlp(ContinualModel):
    """Per-weight metaplasticity (activation-gradient consistency) on an MLP."""

    NAME = 'plasticity_mlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Seq-MNIST protocol of the CLS-ER table: SGD, lr 0.03.
        parser.set_defaults(optimizer='sgd', lr=0.03)

        parser.add_argument('--plast_mlp', type=binary_to_boolean_type, default=0,
                            help='Enable per-weight metaplasticity. Off = plain SGD.')

        # --- Importance and plasticity map ---
        parser.add_argument('--plast_beta', type=float, default=0.9975,
                            help='EMA momentum of the per-weight signal mean M '
                                 '(window 1/(1-beta) steps). Also the death EMA '
                                 'unless --plast_death_beta is set.')
        parser.add_argument('--plast_lambda', type=float, default=160.0,
                            help='Consolidation strength: p = exp(-lambda * A / S).')
        parser.add_argument('--plast_p_min', type=float, default=0.001,
                            help='Lower clamp on p (deepest freeze).')
        parser.add_argument('--plast_p_max', type=float, default=20.0,
                            help='Upper clamp on p (fastest re-plasticized weight).')
        parser.add_argument('--plast_accum_floor', type=float, default=0.0,
                            help='Anti-erosion floor: a weight whose accumulator ever '
                                 'peaked above 0 cannot fall below floor * peak. '
                                 '1 = never erodes; < 0 = off.')
        parser.add_argument('--plast_consol_stiffness', type=float, default=2.0,
                            help='Stiffness exponent kappa. The accumulator step is '
                                 'scaled by (stiff_max^(1-s) * stiff_min^s)^kappa, '
                                 's = 1 - clamp(p, 0, 1). 0 = off.')
        parser.add_argument('--plast_stiff_min', type=float, default=0.01,
                            help='Stiffness gain base for a fully frozen weight.')
        parser.add_argument('--plast_stiff_max', type=float, default=1000.0,
                            help='Stiffness gain base for a fully plastic weight.')
        parser.add_argument('--plast_layers', type=str, default='fc12',
                            choices=['fc12', 'all'],
                            help='Metaplastic layers: fc12 (fc1, fc2) or all (also '
                                 'the classifier).')

        # --- Per-neuron consolidation budget ---
        parser.add_argument('--plast_budget', type=str, default='neuron',
                            choices=['none', 'neuron'],
                            help='neuron: cap each neuron\'s freezing mass '
                                 'sum_i max(1 - p_ji, 0) at budget_frac * fan_in; '
                                 'over-budget rows have all their freezes scaled '
                                 'down. Boosted weights (p > 1) are untouched.')
        parser.add_argument('--plast_budget_frac', type=float, default=0.4,
                            help='Budget as a fraction of fan-in at the start of the '
                                 'run (>= 1 never binds).')
        parser.add_argument('--plast_budget_frac_end', type=float, default=0.8,
                            help='Budget fraction at the end of the run, reached by '
                                 'a linear ramp. < 0 = constant.')

        # --- Classifier (head) ---
        parser.add_argument('--plast_head_consolidate', type=str, default='past_blocks',
                            choices=['off', 'past_blocks'],
                            help='past_blocks: zero the gradient of the classifier '
                                 'rows (weight + bias) of classes from finished '
                                 'tasks; the current task\'s rows stay plastic.')
        # Classifier-only overrides, used under --plast_layers all. Unset = use the
        # fc1/fc2 value.
        parser.add_argument('--plast_lambda_head', type=float, default=None,
                            help='Classifier override for --plast_lambda.')
        parser.add_argument('--plast_lambda_end_head', type=float, default=None,
                            help='Ramp the classifier\'s lambda linearly to this value '
                                 'over the run. Unset or < 0 = constant.')
        parser.add_argument('--plast_budget_head', type=str, default=None,
                            choices=['none', 'neuron'],
                            help='Classifier override for --plast_budget.')
        parser.add_argument('--plast_budget_frac_head', type=float, default=None,
                            help='Classifier override for --plast_budget_frac.')
        parser.add_argument('--plast_budget_frac_end_head', type=float, default=None,
                            help='Classifier override for --plast_budget_frac_end.')
        parser.add_argument('--plast_p_min_head', type=float, default=None,
                            help='Classifier override for --plast_p_min.')
        parser.add_argument('--plast_p_max_head', type=float, default=None,
                            help='Classifier override for --plast_p_max.')

        # --- Dendritic spillover ---
        parser.add_argument('--plast_spread', type=float, default=1.0,
                            help='Fraction of importance replaced by its dendritic '
                                 'neighbourhood average (0 = per-weight). Needs a '
                                 'geometry; ignored with --plast_geometry none.')
        parser.add_argument('--plast_geometry', type=str, default='conductance',
                            choices=['none', 'conductance', 'travel_cost'],
                            help='Dendrite model. conductance: Gaussian resistivity '
                                 'bumps, u = integral of resistivity. travel_cost: '
                                 'discrete barriers, u = x + summed heights crossed. '
                                 'none: no spillover.')
        parser.add_argument('--plast_geo_sigma', type=float, default=0.02,
                            help='Spillover length scale: K = exp(-|du| / sigma).')
        parser.add_argument('--plast_geo_n_barriers', type=int, default=32,
                            help='Barriers per neuron.')
        parser.add_argument('--plast_geo_barrier_height', type=float, default=8.0,
                            help='Max bump amplitude (conductance) / barrier height '
                                 '(travel_cost).')
        parser.add_argument('--plast_geo_barrier_width', type=float, default=0.2,
                            help='Gaussian bump width (conductance only).')
        parser.add_argument('--plast_geo_bins', type=int, default=256,
                            help='Bins per neuron for the binned spillover filter '
                                 '(more = closer to the exact kernel average).')
        parser.add_argument('--plast_geo_seed', type=int, default=0,
                            help='RNG seed for the dendrite layouts (independent of '
                                 '--seed).')

        # --- Learning rate ---
        parser.add_argument('--plast_lr_min', type=float, default=0.0,
                            help='Final LR of the cosine schedule (from --lr) over '
                                 'the whole run.')

        # --- Neuron death ---
        parser.add_argument('--plast_death', type=str, default='weights_line',
                            choices=['off', 'weights_line'],
                            help='weights_line: recycle fc1/fc2 neurons that stay too '
                                 'plastic. Incoming weights re-init (Xavier), bias, '
                                 'importance and dendrite reset, p = 1, outgoing '
                                 'column in the next layer zeroed.')
        parser.add_argument('--plast_death_thresh', type=float, default=0.3,
                            help='Eligible neurons are the top fraction of the layer '
                                 'by EMA mean plasticity.')
        parser.add_argument('--plast_death_window', type=int, default=100,
                            help='Consecutive eligible steps before a neuron dies.')
        parser.add_argument('--plast_death_beta', type=float, default=None,
                            help='EMA momentum of the per-neuron plasticity used by '
                                 'death. Unset = --plast_beta.')
        parser.add_argument('--plast_death_warmup', type=int, default=500,
                            help='No deaths before this many steps.')
        parser.add_argument('--plast_death_rate_cap', type=float, default=0.05,
                            help='Max deaths per step per layer: < 1 = fraction of '
                                 'the layer, >= 1 = count, <= 0 = uncapped. '
                                 'Most-plastic first.')

        # --- Diagnostics (read-only, logged at task boundaries) ---
        parser.add_argument('--plast_ncm_oracle', type=binary_to_boolean_type, default=0,
                            help='Nearest-class-mean accuracy on fc2 features: a '
                                 'head-free upper bound ("plast NCM ORACLE").')
        parser.add_argument('--plast_probe', type=binary_to_boolean_type, default=0,
                            help='Per-layer p mean/std, frozen and fully-plastic '
                                 'fractions ("plast PROBE").')
        parser.add_argument('--plast_geo_probe', type=binary_to_boolean_type, default=0,
                            help='Per-neuron plasticity, dendrite u-range and '
                                 'effective compartments ("plast GEO PROBE").')
        parser.add_argument('--plast_turnover_probe', type=binary_to_boolean_type,
                            default=0,
                            help='Track when each weight first froze; log frozen / '
                                 'newly frozen / thawed fractions ("plast TURNOVER").')
        parser.add_argument('--plast_turnover_path', type=str, default='',
                            help='With the turnover probe: save per-weight maps to '
                                 '<path>.task<T>.pt each task, for '
                                 'scripts/plot_plasticity_turnover.py and '
                                 'scripts/plot_plasticity_weightgrid.py.')

        # --- Replay combined with the brake (at most one; needs --plast_mlp 1) ---
        parser.add_argument('--plast_derpp', type=binary_to_boolean_type, default=0,
                            help='DER++ replay: logit distillation + label replay '
                                 'from a reservoir buffer.')
        parser.add_argument('--plast_derpp_alpha', type=float, default=1.0,
                            help='DER++ logit-distillation (MSE) weight.')
        parser.add_argument('--plast_derpp_beta', type=float, default=0.5,
                            help='DER++ label-replay (CE) weight.')
        parser.add_argument('--plast_derpp_buffer_size', type=int, default=500,
                            help='DER++ buffer size.')
        parser.add_argument('--plast_derpp_minibatch', type=int, default=-1,
                            help='DER++ replay batch size; -1 = --batch_size.')
        parser.add_argument('--plast_clser', type=binary_to_boolean_type, default=0,
                            help='CLS-ER replay: stable + plastic EMA models and a '
                                 'consistency loss. Inference uses the stable model.')
        parser.add_argument('--plast_clser_buffer_size', type=int, default=500,
                            help='CLS-ER buffer size.')
        parser.add_argument('--plast_clser_minibatch', type=int, default=-1,
                            help='CLS-ER replay batch size; -1 = --batch_size.')
        parser.add_argument('--plast_clser_reg_weight', type=float, default=1.0,
                            help='CLS-ER consistency-loss weight.')
        parser.add_argument('--plast_clser_stable_update_freq', type=float, default=0.8,
                            help='Per-step probability of updating the stable model.')
        parser.add_argument('--plast_clser_stable_alpha', type=float, default=0.99,
                            help='EMA decay of the stable model.')
        parser.add_argument('--plast_clser_plastic_update_freq', type=float,
                            default=1.0,
                            help='Per-step probability of updating the plastic model.')
        parser.add_argument('--plast_clser_plastic_alpha', type=float, default=0.99,
                            help='EMA decay of the plastic model.')
        parser.add_argument('--plast_gem', type=binary_to_boolean_type, default=0,
                            help='GEM replay: project the gradient so no past task\'s '
                                 'loss increases (needs `pip install quadprog`).')
        parser.add_argument('--plast_gem_buffer_size', type=int, default=500,
                            help='GEM episodic memory size, split across tasks.')
        parser.add_argument('--plast_gem_gamma', type=float, default=0.5,
                            help='GEM QP margin.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        for attr in ('fc1', 'fc2', 'classifier'):
            assert hasattr(self.net, attr), (
                f"plasticity_mlp needs an mnistmlp backbone with a `{attr}` "
                f"Linear (run with --backbone mnistmlp).")
        self._enabled = bool(args.plast_mlp)

        # Replay only runs together with the brake.
        self._derpp = self._enabled and bool(args.plast_derpp)
        self._clser = self._enabled and bool(args.plast_clser)
        self._gem = self._enabled and bool(args.plast_gem)
        assert sum((self._derpp, self._clser, self._gem)) <= 1, (
            "plasticity_mlp: --plast_derpp / --plast_clser / --plast_gem are "
            "mutually exclusive (pick one replay mechanism).")
        for flag, name in ((args.plast_derpp, 'derpp'), (args.plast_clser, 'clser'),
                           (args.plast_gem, 'gem')):
            if bool(flag) and not self._enabled:
                logging.warning(f"plasticity_mlp: --plast_{name} ignored because "
                                f"--plast_mlp is 0. For replay without the brake use "
                                f"--model {name}.")
        if self._derpp:
            self._buffer = Buffer(args.plast_derpp_buffer_size)
            self._derpp_mb = (args.plast_derpp_minibatch
                              if args.plast_derpp_minibatch > 0 else args.batch_size)
            logging.info(f"plasticity_mlp: DER++ replay ON (alpha="
                         f"{args.plast_derpp_alpha} beta={args.plast_derpp_beta} "
                         f"buffer={args.plast_derpp_buffer_size} "
                         f"minibatch={self._derpp_mb}).")
        elif self._clser:
            self._buffer = Buffer(args.plast_clser_buffer_size)
        elif self._gem:
            self._buffer = Buffer(args.plast_gem_buffer_size)
        else:
            self._buffer = None

        # CLS-ER semantic memories are built in begin_task, once the backbone is on
        # its device.
        self._stable_model = None
        self._plastic_model = None
        self._clser_step = 0
        if self._clser:
            self._clser_mb = (args.plast_clser_minibatch
                              if args.plast_clser_minibatch > 0 else args.batch_size)
            self._clser_consistency = nn.MSELoss(reduction='none')
            logging.info(
                f"plasticity_mlp: CLS-ER replay ON (reg="
                f"{args.plast_clser_reg_weight} stable={args.plast_clser_stable_alpha}"
                f"@{args.plast_clser_stable_update_freq} plastic="
                f"{args.plast_clser_plastic_alpha}@{args.plast_clser_plastic_update_freq}"
                f" buffer={args.plast_clser_buffer_size} minibatch={self._clser_mb}).")

        # GEM: flat gradient buffer (allocated in begin_task) and one reference
        # gradient per finished task (appended in end_task).
        self._gem_grad_dims = [p.data.numel() for p in self.parameters()]
        self._gem_grads_cs = []
        self._gem_grads_da = None
        self._gem_solver = None
        if self._gem:
            logging.info(f"plasticity_mlp: GEM replay ON (buffer="
                         f"{args.plast_gem_buffer_size} gamma={args.plast_gem_gamma}); "
                         f"projection runs before the brake.")

        self._layers = [self.net.fc1, self.net.fc2]
        if args.plast_layers == 'all':
            self._layers.append(self.net.classifier)
        self._is_head = [layer is self.net.classifier for layer in self._layers]
        # The layer fed by each layer's neurons; None for the classifier, whose
        # units never die.
        nexts = {id(self.net.fc1): self.net.fc2, id(self.net.fc2): self.net.classifier}
        self._next_layer = [nexts.get(id(layer)) for layer in self._layers]

        # Per-weight state, allocated in begin_task (after the backbone has moved
        # to its device). One entry per metaplastic layer.
        n = len(self._layers)
        self._plast_ready = False
        self._M = [None] * n           # EMA mean of the signal s
        self._A = [None] * n           # signed importance accumulator
        self._Apeak = [None] * n       # running max of A (anti-erosion floor)
        self._S = [0.0] * n            # cumulative mean importance (normalizes A)
        self._p = [None] * n           # current plasticity map
        self._geo = [None] * n         # DendriteGeometry, or None
        self._P = [None] * n           # EMA per-neuron mean plasticity (death)
        self._death_count = [None] * n  # consecutive steps each neuron was eligible
        self._deaths_this_task = 0
        # Bookkeeping for the turnover probe and dumps: the step / task at which
        # each weight first froze (-1 = never), the frozen mask at the last task
        # boundary, and how many times each weight was re-initialized by death.
        self._freeze_step = [None] * n
        self._freeze_task = [None] * n
        self._frozen_at_boundary = [None] * n
        self._reset_count = [None] * n
        self._task_start_step = []

        # Hooks cache (layer input, pre-activation) only while _caching is True,
        # i.e. during the current-batch forward in observe().
        self._caching = False
        self._cache = [None] * n
        self._observe_step = 0
        self._total_steps = 0
        for i, layer in enumerate(self._layers):
            layer.register_forward_hook(self._make_hook(i))

    # ------------------------------------------------------------------
    # Setup and task boundaries
    # ------------------------------------------------------------------

    def _make_hook(self, i: int):
        def hook(module, inputs, output):
            if self._caching:
                self._cache[i] = (inputs[0].detach(), output.detach())
        return hook

    def set_caching(self, enabled: bool):
        """Turn the activation cache on or off (for external scripts and tests)."""
        self._caching = enabled

    def _setup_plasticity(self, steps_per_epoch: int, n_tasks: int = 1):
        args = self.args
        # Total optimizer steps in the run (budget ramp, LR schedule, head lambda ramp).
        self._total_steps = steps_per_epoch * max(int(args.n_epochs), 1) * max(int(n_tasks), 1)
        for i, layer in enumerate(self._layers):
            w = layer.weight.data                    # (out, in)
            self._M[i] = torch.zeros_like(w)
            self._A[i] = torch.zeros_like(w)
            self._Apeak[i] = torch.zeros_like(w)
            self._S[i] = 0.0
            self._p[i] = torch.ones_like(w)
            self._P[i] = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
            self._death_count[i] = torch.zeros(w.shape[0], device=w.device, dtype=torch.long)
            self._freeze_step[i] = torch.full_like(w, -1, dtype=torch.long)
            self._freeze_task[i] = torch.full_like(w, -1, dtype=torch.long)
            self._frozen_at_boundary[i] = torch.zeros_like(w, dtype=torch.bool)
            self._reset_count[i] = torch.zeros_like(w, dtype=torch.long)
            if args.plast_geometry != 'none':
                # Per-layer seed so fc1 and fc2 get different layouts.
                self._geo[i] = DendriteGeometry(
                    w.shape[0], w.shape[1], geometry=args.plast_geometry,
                    n_barriers=args.plast_geo_n_barriers,
                    barrier_height=args.plast_geo_barrier_height,
                    barrier_width=args.plast_geo_barrier_width,
                    seed=args.plast_geo_seed + 1000 * i,
                    device=w.device, dtype=w.dtype)
        self._plast_ready = True
        logging.info(
            f"plasticity_mlp: beta={args.plast_beta} death_beta={self._death_beta()} "
            f"lambda={args.plast_lambda} p_in[{args.plast_p_min},{args.plast_p_max}] "
            f"floor={args.plast_accum_floor} stiffness={args.plast_consol_stiffness}"
            f"[{args.plast_stiff_min},{args.plast_stiff_max}] layers={args.plast_layers} "
            f"budget={args.plast_budget}({args.plast_budget_frac}->{args.plast_budget_frac_end}) "
            f"spread={args.plast_spread} geometry={args.plast_geometry} "
            f"death={args.plast_death} head={args.plast_head_consolidate} "
            f"total_steps={self._total_steps}.")

    def begin_task(self, dataset):
        if self._enabled and not self._plast_ready:
            self._setup_plasticity(steps_per_epoch=len(dataset.train_loader),
                                   n_tasks=getattr(dataset, 'N_TASKS', 1))
        if self._clser and self._stable_model is None:
            self._plastic_model = self._make_ema_copy()
            self._stable_model = self._make_ema_copy()
        if self._gem and self._gem_grads_da is None:
            self._gem_grads_da = torch.zeros(sum(self._gem_grad_dims), device=self.device)
        self._deaths_this_task = 0
        if self._enabled and self.args.plast_turnover_probe:
            self._task_start_step.append(self._observe_step)

    @torch.no_grad()
    def end_task(self, dataset):
        if not self._enabled:
            return
        if self._gem:
            # Reserve this task's reference-gradient slot and store its share of the
            # episodic memory. fill_buffer reads args.buffer_size, so point it at
            # --plast_gem_buffer_size.
            self._gem_grads_cs.append(
                torch.zeros(sum(self._gem_grad_dims), device=self.device))
            dataset.args.buffer_size = self.args.plast_gem_buffer_size
            fill_buffer(self._buffer, dataset, self.current_task,
                        required_attributes=['examples', 'labels', 'task_labels'])
        if self.args.plast_head_consolidate == 'past_blocks':
            s = self.get_offsets(self.current_task)[0]
            logging.info(f"plast HEAD CONSOLIDATE task {self.current_task}: froze "
                         f"rows [0,{s}) of {self.num_classes}")
        if self.args.plast_death != 'off':
            names = _LAYER_NAMES[:len(self._layers)]
            pstats = ' '.join(
                f"{names[i]}P={float(self._P[i].mean()):.3f}"
                for i in range(len(self._layers)) if self._next_layer[i] is not None)
            logging.info(f"plast DEATH task {self.current_task}: "
                         f"deaths={self._deaths_this_task} | {pstats}")
        if self.args.plast_geo_probe:
            self._run_geo_probe()
        if self.args.plast_turnover_probe:
            self._run_turnover_probe()
        if self.args.plast_ncm_oracle:
            self._run_ncm_oracle(dataset)
        if self.args.plast_probe:
            self._run_plast_probe()

    # ------------------------------------------------------------------
    # Diagnostics (read-only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_ncm_oracle(self, dataset):
        """Nearest-class-mean accuracy on fc2 features over all seen test sets, with
        class means from the current backbone. A head-free upper bound: if this is
        also low, the features were overwritten, not just the readout."""
        was_training = self.net.training
        self.net.eval()
        feats_all, labels_all, tasks_all = [], [], []
        try:
            for t, loader in enumerate(dataset.test_loaders):
                for data in loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    feats_all.append(self.net(inputs, returnt='features').float())
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
            per_task.append(100.0 * int(cil_hit[sel].sum()) / max(int(sel.sum()), 1))
        raw = ', '.join(f'{a:.2f}' for a in per_task)
        logging.info(
            f"plast NCM ORACLE task {self.current_task}: "
            f"[Class-IL] {cil:.2f}%  [Task-IL] {til:.2f}%  | per-task CIL [{raw}]")
        return {'class_il': cil, 'task_il': til, 'per_task': per_task}

    @torch.no_grad()
    def _run_plast_probe(self):
        """Per layer: mean/std of p, frozen fraction (p < 0.1) and fully-plastic
        fraction (p >= 1)."""
        msgs, stats = [], []
        names = _LAYER_NAMES[:len(self._layers)]
        for i in range(len(self._layers)):
            p = self._compute_plasticity(i)
            frozen = float((p < _FREEZE_THRESH).float().mean())
            plastic = float((p >= 1.0).float().mean())
            st = {'layer': names[i], 'mean': float(p.mean()), 'std': float(p.std()),
                  'frozen': frozen, 'plastic': plastic}
            stats.append(st)
            msgs.append(f"{names[i]}: mean={st['mean']:.3f} std={st['std']:.3f} "
                        f"frozen={frozen:.3f} plastic={plastic:.3f}")
        logging.info(f"plast PROBE task {self.current_task}: " + " | ".join(msgs))
        return stats

    @torch.no_grad()
    def _run_geo_probe(self):
        """Per layer: per-neuron mean plasticity P_j = mean_i p_ji (mean/std),
        barriers per neuron, mean dendrite u-range, and effective compartments per
        neuron (1 + number of gaps between neighbouring synapses in u wider than
        geo_sigma)."""
        names = _LAYER_NAMES[:len(self._layers)]
        sigma = max(float(self.args.plast_geo_sigma), _EPS)
        stats, msgs = [], []
        for i in range(len(self._layers)):
            P = self._p[i].mean(dim=1)
            st = {'layer': names[i], 'P_mean': float(P.mean()),
                  'P_std': float(P.std(unbiased=False))}
            geo = self._geo[i]
            if geo is not None:
                Us, _ = geo.U.sort(dim=1)
                rng = Us[:, -1] - Us[:, 0]
                breaks = (Us[:, 1:] - Us[:, :-1] > sigma).sum(dim=1)
                st.update({'barriers_per_neuron': geo.n_barriers,
                           'u_range_mean': float(rng.mean()),
                           'eff_compartments': float((breaks.float() + 1.0).mean())})
                msgs.append(
                    f"{names[i]}: P={st['P_mean']:.3f}±{st['P_std']:.3f} "
                    f"barriers={geo.n_barriers} u_range={st['u_range_mean']:.2f} "
                    f"eff_comp={st['eff_compartments']:.2f}")
            else:
                msgs.append(f"{names[i]}: P={st['P_mean']:.3f}±{st['P_std']:.3f} "
                            f"(no geometry)")
            stats.append(st)
        logging.info(f"plast GEO PROBE task {self.current_task}: "
                     f"deaths={self._deaths_this_task} | " + " | ".join(msgs))
        return stats

    @torch.no_grad()
    def _run_turnover_probe(self):
        """Freeze turnover per layer at a task boundary:

          frozen     fraction with p < 0.1 now
          ever       fraction that has ever frozen
          new        fraction that first froze during this task
          thaw       fraction frozen at the previous boundary but not now
          p_froz     mean p of frozen weights
          p_plas     mean p of the rest
          p10/50/90  percentiles of p
          when       mean position within this task (0 = start, 1 = end) of this
                     task's first freezes

        With --plast_turnover_path, also saves the per-weight maps for the plot
        scripts, one file per task."""
        names = _LAYER_NAMES[:len(self._layers)]
        t = self.current_task
        t_start = self._task_start_step[t] if t < len(self._task_start_step) else 0
        t_len = max(self._observe_step - t_start, 1)
        stats, msgs = [], []
        for i in range(len(self._layers)):
            p = self._p[i]
            n = p.numel()
            frozen_now = p < _FREEZE_THRESH
            ever = self._freeze_step[i] >= 0
            new = self._freeze_task[i] == t
            thawed = self._frozen_at_boundary[i] & ~frozen_now
            n_froz = int(frozen_now.sum())
            n_plas = n - n_froz
            p_froz = float(p[frozen_now].mean()) if n_froz else 0.0
            p_plas = float(p[~frozen_now].mean()) if n_plas else 0.0
            q = torch.quantile(
                p.flatten().float(),
                torch.tensor([0.1, 0.5, 0.9], device=p.device, dtype=torch.float))
            fresh_steps = self._freeze_step[i][new]
            when = (float((fresh_steps.float() - t_start).mean()) / t_len
                    if fresh_steps.numel() else float('nan'))
            st = {'layer': names[i], 'frozen': n_froz / n, 'ever': int(ever.sum()) / n,
                  'new': int(new.sum()) / n, 'thaw': int(thawed.sum()) / n,
                  'p_frozen': p_froz, 'p_plastic': p_plas,
                  'p10': float(q[0]), 'p50': float(q[1]), 'p90': float(q[2]),
                  'when': when}
            stats.append(st)
            msgs.append(
                f"{names[i]}: frozen={st['frozen']:.3f} ever={st['ever']:.3f} "
                f"new={st['new']:.3f} thaw={st['thaw']:.3f} "
                f"p_froz={p_froz:.3f} p_plas={p_plas:.3f} "
                f"p[.1/.5/.9]={st['p10']:.2f}/{st['p50']:.2f}/{st['p90']:.2f} "
                f"when={when:.2f}")
            self._frozen_at_boundary[i] = frozen_now.clone()
        logging.info(f"plast TURNOVER task {t}: " + " | ".join(msgs))

        path = self.args.plast_turnover_path
        if path:
            base, ext = os.path.splitext(path)
            out = f"{base}.task{t}{ext or '.pt'}"
            n_layers = len(self._layers)
            torch.save({
                'layers': names,
                'freeze_step': [self._freeze_step[i].cpu() for i in range(n_layers)],
                'freeze_task': [self._freeze_task[i].cpu() for i in range(n_layers)],
                'p': [self._p[i].detach().cpu() for i in range(n_layers)],
                'reset_count': [self._reset_count[i].cpu() for i in range(n_layers)],
                'p_range': [(float(self._arg(i, 'plast_p_min')),
                             float(self._arg(i, 'plast_p_max')))
                            for i in range(n_layers)],
                'death_mode': self.args.plast_death,
                'task_start_step': list(self._task_start_step),
                'total_steps_so_far': self._observe_step,
                'current_task': t,
                'n_tasks_seen': len(self._task_start_step),
                'freeze_thresh': _FREEZE_THRESH,
            }, out)
            logging.info(f"plast TURNOVER task {t}: dumped maps -> {out}")
        return stats

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_mean(a_prev, z):
        """Batch mean of s_ji = a_prev_i * 1[z_j > 0], shape (out, in). The signal
        factorizes, so this is one matmul instead of a per-sample (out, in) tensor."""
        return ((z > 0).to(z.dtype).t() @ a_prev) / a_prev.shape[0]

    @torch.no_grad()
    def _update_stats(self):
        """EMA-update M from the activations cached in this step's forward."""
        beta = self.args.plast_beta
        for i in range(len(self._layers)):
            if self._cache[i] is None:
                continue
            a_prev, z = self._cache[i]
            self._M[i].mul_(beta).add_(self._signal_mean(a_prev, z), alpha=1.0 - beta)

    def _omega(self, i: int) -> torch.Tensor:
        """Importance Omega = |M| after dendritic spillover. |M| rather than M
        because fc1's input (normalized pixels) is signed."""
        return self._spread(i, self._M[i].abs())

    def _spread(self, i: int, Omega: torch.Tensor) -> torch.Tensor:
        """Mix each weight's importance with its dendritic neighbourhood average,
        nbr_ji = sum_k K_j(i,k) Omega_jk / sum_k K_j(i,k), K_j = exp(-|u_ji - u_jk| /
        sigma), then rescale to the original layer mean so lambda and S stay
        calibrated. Identity when --plast_spread is 0 or there is no geometry."""
        a = self.args.plast_spread
        geo = self._geo[i]
        if a <= 0.0 or geo is None:
            return Omega
        nbr = geo.spread_binned(Omega, self.args.plast_geo_sigma, self.args.plast_geo_bins)
        out = (1.0 - a) * Omega + a * nbr
        # Guard on an exactly-zero mean instead of adding eps: importance is tiny in
        # absolute terms, so an eps in the denominator would bias the scale.
        denom = out.mean()
        if float(denom) > 0.0:
            out = out * (Omega.mean() / denom)
        return out

    @torch.no_grad()
    def _accumulate(self, i: int):
        """A += G * (Omega - mean(Omega)), S += mean(Omega), then the floor.

        G = (stiff_max^(1-s) * stiff_min^s)^kappa with frozen-ness s = 1 - clamp(p,
        0, 1) from the previous step: frozen weights barely move their accumulator,
        plastic ones move fast. The floor stops a weight whose A ever peaked above
        0 from eroding below floor * peak; never-important weights may go negative
        (and so get p > 1)."""
        Omega = self._omega(i)
        mean = Omega.mean()
        delta = Omega - mean
        kappa = self.args.plast_consol_stiffness
        if kappa > 0.0:
            s = 1.0 - self._p[i].clamp(0.0, 1.0)
            log_g = ((1.0 - s) * math.log(max(self.args.plast_stiff_max, _EPS))
                     + s * math.log(max(self.args.plast_stiff_min, _EPS)))
            delta = delta * torch.exp(kappa * log_g)
        self._A[i] += delta
        self._S[i] += float(mean)
        torch.maximum(self._Apeak[i], self._A[i], out=self._Apeak[i])
        floor = self.args.plast_accum_floor
        if floor >= 0.0:
            lo = floor * self._Apeak[i]
            self._A[i] = torch.where(self._Apeak[i] > _EPS,
                                     torch.maximum(self._A[i], lo), self._A[i])

    # ------------------------------------------------------------------
    # Plasticity map
    # ------------------------------------------------------------------

    def _compute_plasticity(self, i: int) -> torch.Tensor:
        """p = clamp(exp(-lambda * A / S), p_min, p_max), then the budget. Reads
        state only (the probes call it too)."""
        Z = self._A[i] / (self._S[i] + _EPS)
        expo = (-self._lambda_eff(i) * Z).clamp(-50.0, 50.0)
        p = torch.exp(expo).clamp(self._arg(i, 'plast_p_min'), self._arg(i, 'plast_p_max'))
        return self._apply_budget(i, p)

    def _arg(self, i: int, name: str):
        """Hyperparameter `name` for layer i: the `<name>_head` override for the
        classifier when it is set, else the shared value."""
        if self._is_head[i]:
            override = getattr(self.args, name + '_head', None)
            if override is not None:
                return override
        return getattr(self.args, name)

    def _progress(self) -> float:
        """Fraction of the run completed, in [0, 1]."""
        return min(self._observe_step / self._total_steps, 1.0)

    def _lambda_eff(self, i: int) -> float:
        """Lambda for layer i. Constant, except that the classifier ramps linearly
        to --plast_lambda_end_head over the run when that is set."""
        start = self._arg(i, 'plast_lambda')
        end = self.args.plast_lambda_end_head if self._is_head[i] else None
        if end is None or end < 0.0 or self._total_steps <= 0:
            return start
        return start + (end - start) * self._progress()

    def _budget_frac_eff(self, i: int) -> float:
        """Budget fraction for layer i, ramped linearly from --plast_budget_frac to
        --plast_budget_frac_end over the run (constant if the end is < 0)."""
        start = self._arg(i, 'plast_budget_frac')
        end = self._arg(i, 'plast_budget_frac_end')
        if end < 0.0 or self._total_steps <= 0:
            return start
        return start + (end - start) * self._progress()

    def _apply_budget(self, i: int, p: torch.Tensor) -> torch.Tensor:
        """Per-neuron consolidation budget. A weight's freezing mass is c = max(1 -
        p, 0). If a neuron's (row's) total exceeds budget_frac * fan_in, all of its
        c are scaled down so the total equals the budget: p <- 1 - c * scale. Rows
        within budget and boosted weights (p >= 1) are unchanged."""
        if self._arg(i, 'plast_budget') != 'neuron':
            return p
        c = (1.0 - p).clamp_min(0.0)
        budget = self._budget_frac_eff(i) * p.shape[1]
        scale = (budget / (c.sum(dim=1, keepdim=True) + _EPS)).clamp(max=1.0)
        return torch.where(p < 1.0, 1.0 - c * scale, p)

    # ------------------------------------------------------------------
    # Training step pieces
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_plasticity(self):
        """Update the importance state and scale each metaplastic weight gradient by
        its plasticity (biases are not gated)."""
        self._update_stats()
        for i in range(len(self._layers)):
            self._accumulate(i)
        for i, layer in enumerate(self._layers):
            p = self._compute_plasticity(i)
            self._p[i] = p
            if layer.weight.grad is not None:
                layer.weight.grad.mul_(p)
        if self.args.plast_turnover_probe:
            self._update_turnover()
        self._consolidate_head()
        self._cache = [None] * len(self._layers)

    @torch.no_grad()
    def _consolidate_head(self):
        """--plast_head_consolidate past_blocks: zero the classifier gradient rows
        (weight + bias) of classes from finished tasks. Runs after the gating, so
        under --plast_layers all it overrides p on those rows."""
        if self.args.plast_head_consolidate != 'past_blocks':
            return
        s = self.get_offsets(self.current_task)[0]
        if s <= 0:
            return
        clf = self.net.classifier
        if clf.weight.grad is not None:
            clf.weight.grad[:s] = 0.0
        if clf.bias is not None and clf.bias.grad is not None:
            clf.bias.grad[:s] = 0.0

    @torch.no_grad()
    def _update_turnover(self):
        """Record the first step and task at which each weight froze. Later thaws and
        re-freezes do not overwrite it."""
        for i in range(len(self._layers)):
            newly = (self._p[i] < _FREEZE_THRESH) & (self._freeze_step[i] < 0)
            if bool(newly.any()):
                self._freeze_step[i][newly] = self._observe_step
                self._freeze_task[i][newly] = self.current_task

    @torch.no_grad()
    def _apply_lr_schedule(self):
        """Cosine-anneal the LR from --lr to --plast_lr_min over the whole run. Set
        every step, so it survives the per-task optimizer reset."""
        if self._total_steps <= 0:
            return
        lr_min = self.args.plast_lr_min
        lr = lr_min + 0.5 * (self.args.lr - lr_min) * (1.0 + math.cos(math.pi * self._progress()))
        for g in self.opt.param_groups:
            g['lr'] = lr

    # ------------------------------------------------------------------
    # Neuron death
    # ------------------------------------------------------------------

    def _death_beta(self) -> float:
        b = self.args.plast_death_beta
        return self.args.plast_beta if b is None else b

    def _death_cap(self, out_features: int) -> int:
        """Max deaths this step: a fraction of the layer (cap < 1, at least 1), an
        absolute count (cap >= 1), or everything (cap <= 0)."""
        rc = self.args.plast_death_rate_cap
        if rc <= 0.0:
            return out_features
        if rc < 1.0:
            return max(1, int(rc * out_features))
        return int(rc)

    @torch.no_grad()
    def _maybe_kill_neurons(self):
        """After the optimizer step: EMA each neuron's mean plasticity P_j. Neurons in
        the layer's top --plast_death_thresh fraction by P for --plast_death_window
        consecutive steps die, most plastic first, at most the rate cap per step."""
        if self.args.plast_death != 'weights_line':
            return
        beta = self._death_beta()
        for i in range(len(self._layers)):
            if self._next_layer[i] is None:
                continue
            self._P[i].mul_(beta).add_(self._p[i].mean(dim=1), alpha=1.0 - beta)
            if self._observe_step < self.args.plast_death_warmup:
                continue
            P = self._P[i]
            q = torch.quantile(P, min(max(1.0 - self.args.plast_death_thresh, 0.0), 1.0))
            above = P >= q
            self._death_count[i][above] += 1
            self._death_count[i][~above] = 0
            eligible = (self._death_count[i] >= self.args.plast_death_window
                        ).nonzero(as_tuple=True)[0]
            if eligible.numel() == 0:
                continue
            cap = self._death_cap(self._layers[i].weight.shape[0])
            if eligible.numel() > cap:
                order = torch.argsort(P[eligible], descending=True)
                eligible = eligible[order[:cap]]
            self._kill(i, eligible)
            self._deaths_this_task += int(eligible.numel())

    @torch.no_grad()
    def _kill(self, i: int, rows: torch.Tensor):
        """Re-initialize neurons `rows` of layer i: new dendrite layout, Xavier
        incoming weights, zero bias and importance, p = 1, and a zeroed outgoing
        column in the next layer so the reborn unit does not jolt downstream."""
        layer = self._layers[i]
        if self._geo[i] is not None:
            self._geo[i].roll(rows)
        self._death_count[i][rows] = 0
        self._P[i][rows] = 0.0
        new = torch.empty_like(layer.weight.data)
        torch.nn.init.xavier_uniform_(new)
        layer.weight.data[rows] = new[rows]
        if layer.bias is not None:
            layer.bias.data[rows] = 0.0
        for buf in (self._M[i], self._A[i], self._Apeak[i]):
            buf[rows] = 0.0
        self._p[i][rows] = 1.0
        self._reset_count[i][rows] += 1
        self._freeze_step[i][rows] = -1
        self._freeze_task[i][rows] = -1
        self._frozen_at_boundary[i][rows] = False
        self._next_layer[i].weight.data[:, rows] = 0.0

    # ------------------------------------------------------------------
    # CLS-ER replay
    # ------------------------------------------------------------------

    def _make_ema_copy(self):
        """A frozen copy of the working net without its forward hooks, so it never
        writes to the activation cache."""
        ema = deepcopy(self.net).to(self.device)
        for module in ema.modules():
            module._forward_hooks.clear()
        for param in ema.parameters():
            param.requires_grad_(False)
        return ema

    @torch.no_grad()
    def _update_ema_models(self):
        """After the step, EMA the working net into each memory with its own
        per-step probability (as in CLS-ER)."""
        self._clser_step += 1
        for model, freq, a in (
                (self._plastic_model, self.args.plast_clser_plastic_update_freq,
                 self.args.plast_clser_plastic_alpha),
                (self._stable_model, self.args.plast_clser_stable_update_freq,
                 self.args.plast_clser_stable_alpha)):
            if float(torch.rand(1)) < freq:
                alpha = min(1 - 1 / (self._clser_step + 1), a)
                for ema_p, p in zip(model.parameters(), self.net.parameters()):
                    ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    def _clser_loss(self, inputs, labels):
        """CLS-ER loss: CE on current + buffer samples plus a consistency MSE toward
        whichever memory is more confident on the true label. Only the current-batch
        forward fills the activation cache. Using one buffer forward for both terms
        gives the same gradient as stock CLS-ER's two forwards."""
        self._caching = True
        try:
            cur_outputs = self.net(inputs)
        finally:
            self._caching = False
        if self._buffer.is_empty():
            return self.loss(cur_outputs, labels)

        buf_inputs, buf_labels = self._buffer.get_data(
            self._clser_mb, transform=self.transform, device=self.device)
        with torch.no_grad():
            stable_logits = self._stable_model(buf_inputs)
            plastic_logits = self._plastic_model(buf_inputs)
        label_mask = F.one_hot(buf_labels, num_classes=stable_logits.shape[-1]) > 0
        sel = (F.softmax(stable_logits, 1)[label_mask]
               > F.softmax(plastic_logits, 1)[label_mask]).unsqueeze(1)
        ema_logits = torch.where(sel, stable_logits, plastic_logits)

        buf_outputs = self.net(buf_inputs)
        l_cons = torch.mean(self._clser_consistency(buf_outputs, ema_logits.detach()))
        ce = self.loss(torch.cat((cur_outputs, buf_outputs)),
                       torch.cat((labels, buf_labels)))
        return ce + self.args.plast_clser_reg_weight * l_cons

    def forward(self, x):
        """Under CLS-ER, inference uses the stable memory; otherwise the working net."""
        if self._clser and self._stable_model is not None:
            return self._stable_model(x)
        return self.net(x)

    # ------------------------------------------------------------------
    # GEM replay
    # ------------------------------------------------------------------

    def _gem_solver_once(self):
        if self._gem_solver is None:
            try:
                import quadprog as solver
            except ImportError as e:
                raise RuntimeError(
                    "--plast_gem requires the `quadprog` QP solver "
                    "(pip install quadprog).") from e
            self._gem_solver = solver
        return self._gem_solver

    def _gem_reference_grads(self):
        """Store each past task's gradient on its memory in grads_cs. Leaves .grad
        dirty; the caller zeroes it before the current forward."""
        buf_inputs, buf_labels, buf_tl = self._buffer.get_data(
            self.args.plast_gem_buffer_size, transform=self.transform,
            device=self.device)
        for tt in buf_tl.unique():
            self.opt.zero_grad()
            sel = buf_tl == tt
            penalty = self.loss(self.net(buf_inputs[sel]), buf_labels[sel])
            penalty.backward()
            store_grad(self.parameters, self._gem_grads_cs[int(tt)], self._gem_grad_dims)

    @torch.no_grad()
    def _gem_project(self):
        """If the current gradient would increase any past task's loss, overwrite it
        with its GEM projection."""
        if self._buffer.is_empty() or not self._gem_grads_cs:
            return
        store_grad(self.parameters, self._gem_grads_da, self._gem_grad_dims)
        G = torch.stack(self._gem_grads_cs).T            # (n_params, n_past_tasks)
        dot = torch.mm(self._gem_grads_da.unsqueeze(0), G)
        if (dot < 0).sum() != 0:
            project2cone2(self._gem_solver_once(), self._gem_grads_da.unsqueeze(1),
                          G, margin=self.args.plast_gem_gamma)
            overwrite_grad(self.parameters, self._gem_grads_da, self._gem_grad_dims)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        if not self._enabled:                    # plain SGD (models/sgd.py)
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.step()
            return loss.item()

        assert self._plast_ready, "begin_task must run before observe"

        self.opt.zero_grad()
        if self._clser:
            loss = self._clser_loss(inputs, labels)
            outputs = None
        else:
            # GEM reference gradients come first, with caching off.
            if self._gem and not self._buffer.is_empty():
                self._gem_reference_grads()
                self.opt.zero_grad()
            self._caching = True
            try:
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels)
            finally:
                self._caching = False
            # DER++ terms add to the loss, so the brake scales their gradient too.
            if self._derpp and not self._buffer.is_empty():
                buf_inputs, _, buf_logits = self._buffer.get_data(
                    self._derpp_mb, transform=self.transform, device=self.device)
                loss = loss + self.args.plast_derpp_alpha * F.mse_loss(
                    self.net(buf_inputs), buf_logits)
                buf_inputs, buf_labels, _ = self._buffer.get_data(
                    self._derpp_mb, transform=self.transform, device=self.device)
                loss = loss + self.args.plast_derpp_beta * self.loss(
                    self.net(buf_inputs), buf_labels)

        loss.backward()
        if self._gem:
            self._gem_project()                  # project first; the brake scales the result
        self._observe_step += 1
        self._apply_plasticity()
        self._apply_lr_schedule()
        self.opt.step()
        if self._derpp:
            self._buffer.add_data(examples=not_aug_inputs, labels=labels,
                                  logits=outputs.data)
        if self._clser:
            self._buffer.add_data(examples=not_aug_inputs, labels=labels)
            self._update_ema_models()
        self._maybe_kill_neurons()
        return loss.item()
