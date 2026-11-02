"""
ABLATION: SOFTEN the brake (lambda) and the consolidation budget in the DER++ x
metaplasticity combo, 2026-07. Follow-up to gen_derpp_plast_sweep.py.

HYPOTHESIS: the brake center used by the replay arms is the sweep13 WINNER, which was
tuned WITHOUT replay (lambda 160, budget 0.4 -> 0.80). Replay and consolidation both
fight forgetting, so once DER++ is bolted on, that much consolidation may be
OVER-rigid -- the brake freezes weights the replay stream would have kept calibrated,
costing plasticity on new tasks for protection replay already provides. If so, the
combo should IMPROVE as the brake is softened, and the best derpp_plast point sits at
a LOWER lambda / SMALLER budget than the no-replay winner.

Axes (everything not named is held at the winner center, so each run is attributable):

  center         the winner brake + DER++ -- the REFERENCE every other point is read
                 against. Also emitted as rung 0 of `joint`; identical lines dedupe.
  lambda_start   --plast_lambda            80, 40           (center 160)
                 Brake strength: p = 1/(1 + lambda*Omega_hat). Lower = weaker brake,
                 more plastic.
  budget_start   --plast_budget_frac       0.3, 0.1         (center 0.4)
                 Per-neuron freezing budget as a fraction of fan-in; the ramp END stays
                 at the center 0.80. Lower = scarcer freezing resources.
  budget_end     (start -> end) pairs      0.4->0.7, 0.4->0.6, 0.2->0.7, 0.2->0.4
                 The budget RAMP given as explicit --plast_budget_frac ->
                 --plast_budget_frac_end pairs (the center ramp is 0.4 -> 0.80). The
                 first two hold the center's start and only lower where the ramp LANDS;
                 the last two also start scarcer.
  joint          lambda + budget lowered together as coherent "softer brake" rungs --
                 the dials interact (both set rigidity), so one-factor-at-a-time alone
                 can miss the win.
  corners        two reference extremes: budget OFF (--plast_budget none) and brake OFF
                 (--plast_lambda 0 => p == 1 everywhere, i.e. replay + the machinery but
                 no consolidation). `brake_off` is the key control: if it matches the
                 best softened point, the brake contributes nothing on top of replay.

There is deliberately NO lambda-ramp axis (--plast_lambda_end stays at the center -1.0,
i.e. constant lambda) -- lambda is probed by its start value only.

PROTOCOL and the DER++ hyperparameters are IDENTICAL to gen_derpp_plast_sweep.py
(buffer 500, alpha 1.0, beta 0.5, minibatch 10), so these runs drop straight into the
same table as the derpp_plast / derpp_plast_head arms. Read mean +/- std over seeds
TOGETHER -- single-seed CIL here is a task-3 coin flip (std 3-6).

Self-contained (no sibling imports) -- copy just this file to the OSC checkout.
Requires models/plasticity_mlp.py with the --plast_derpp* flags AND no stale sibling
in models/ claiming NAME='plasticity_mlp' (a `plasticity_mlp_noderpp.py` once hijacked
the registry and silently ran the pre-derpp code -- verify with:
  python -c "from argparse import Namespace; from models import get_model_class; \
             import sys; c=get_model_class(Namespace(model='plasticity-mlp')); \
             print(sys.modules[c.__module__].__file__)")

Usage:
    python gen_derpp_plast_ablate_sweep.py                    # all axes, 10 seeds
    python gen_derpp_plast_ablate_sweep.py --axes joint corners
    python gen_derpp_plast_ablate_sweep.py --heads off past_blocks
    python gen_derpp_plast_ablate_sweep.py --with_noreplay    # matched brake-only twins
    python gen_derpp_plast_ablate_sweep.py --seeds 0 1 2 3 4  # half-cost scout pass
    sbatch --array=1-$(wc -l < derpp_plast_ablate_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_plast_ablate_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

# DER++ side, identical to the parent sweep so the arms are comparable.
DERPP = dict(buffer_size=500, minibatch=10, alpha=1.0, beta=0.5)

# The sweep13 WINNER brake center (CIL 27.1 / TIL 93.2 with past_blocks), verbatim.
WINNER_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='fc12',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='weights_line', plast_death_thresh=0.3, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.80,
)

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# Ablation axes: name -> [(label, {flag: value, ...}), ...]. Each entry overrides the
# winner center. Multi-flag entries are fine (budget_end moves BOTH ramp ends).
AXES = {
    'lambda_start': [
        ('lambda=80', dict(plast_lambda=80.0)),
        ('lambda=40', dict(plast_lambda=40.0)),
    ],
    'budget_start': [
        ('budget_frac=0.3', dict(plast_budget_frac=0.3)),
        ('budget_frac=0.1', dict(plast_budget_frac=0.1)),
    ],
    # Budget RAMP as explicit (start -> end) pairs.
    'budget_end': [
        ('budget=0.4->0.7', dict(plast_budget_frac=0.4, plast_budget_frac_end=0.7)),
        ('budget=0.4->0.6', dict(plast_budget_frac=0.4, plast_budget_frac_end=0.6)),
        ('budget=0.2->0.7', dict(plast_budget_frac=0.2, plast_budget_frac_end=0.7)),
        ('budget=0.2->0.4', dict(plast_budget_frac=0.2, plast_budget_frac_end=0.4)),
    ],
}

# Coherent joint rungs: (lambda, budget_frac, budget_frac_end). Row 0 == the center,
# so it dedupes against the explicit `center` point.
JOINT = [
    (160.0, 0.4, 0.80),
    (80.0, 0.2, 0.40),
    (40.0, 0.1, 0.20),
    (20.0, 0.05, 0.10),
]

# Reference extremes. brake_off (lambda 0 => p == 1) is replay + machinery, no
# consolidation; budget_off removes the per-neuron freezing cap entirely.
CORNERS = {
    'budget_off': dict(plast_budget='none'),
    'brake_off': dict(plast_lambda=0.0),
}

ALL_AXES = ['center'] + list(AXES) + ['joint', 'corners']


def cfg_to_flags(cfg):
    # Emit NEGATIVE values as `--flag=value` so a bare "-1.0" is never parsed as a flag
    # token (mammoth's check_multiple_defined_arg_during_string_parse treats any
    # '-'-prefixed token as a flag name).
    out = []
    for k, v in cfg.items():
        s = str(v)
        out.append(f'--{k}={s}' if s.startswith('-') else f'--{k} {s}')
    return ' '.join(out)


def plast_line(cfg, head, derpp, seed):
    """One plasticity_mlp run from a brake cfg, optionally with DER++ replay."""
    cfg = dict(cfg)
    cfg['plast_head_consolidate'] = head
    if derpp:
        cfg['plast_derpp'] = 1
        cfg['plast_derpp_buffer_size'] = DERPP['buffer_size']
        cfg['plast_derpp_minibatch'] = DERPP['minibatch']
        cfg['plast_derpp_alpha'] = DERPP['alpha']
        cfg['plast_derpp_beta'] = DERPP['beta']
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def build_configs(selected):
    """(label, cfg) pairs for every requested ablation point."""
    out = []
    for name in selected:
        if name == 'center':
            out.append(('center', dict(WINNER_CENTER)))
        elif name in AXES:
            for label, over in AXES[name]:
                cfg = dict(WINNER_CENTER)
                cfg.update(over)
                out.append((label, cfg))
        elif name == 'joint':
            for lam, bf, bfe in JOINT:
                cfg = dict(WINNER_CENTER)
                cfg['plast_lambda'] = lam
                cfg['plast_budget_frac'] = bf
                cfg['plast_budget_frac_end'] = bfe
                out.append((f'joint=l{lam}/b{bf}->{bfe}', cfg))
        elif name == 'corners':
            for label, over in CORNERS.items():
                cfg = dict(WINNER_CENTER)
                cfg.update(over)
                out.append((label, cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--axes', nargs='+', choices=ALL_AXES, default=None, metavar='AXIS',
                    help='Which ablation axes to emit (default: all). Choices: '
                         + ', '.join(ALL_AXES) + '.')
    ap.add_argument('--heads', nargs='+', choices=['off', 'past_blocks'],
                    default=['past_blocks'], metavar='HEAD',
                    help='--plast_head_consolidate settings (default: past_blocks, the '
                         'sweep13 winner). Pass both to A/B the head freeze.')
    ap.add_argument('--with_noreplay', action='store_true',
                    help='Also emit each config WITHOUT DER++ (matched brake-only twin), '
                         'so the replay DELTA is measurable at every softened setting. '
                         'Doubles the run count.')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_plast_ablate_sweep.txt'))
    args = ap.parse_args()

    selected = args.axes or ALL_AXES
    configs = build_configs(selected)

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line)
            lines.append(line)

    for seed in args.seeds:                    # seed-outer, config-inner
        for head in args.heads:
            for _label, cfg in configs:
                add(plast_line(cfg, head, True, seed))
                if args.with_noreplay:
                    add(plast_line(cfg, head, False, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Unique configs = lines per seed (the center repeats across axes and dedupes).
    uniq = len(lines) // max(len(args.seeds), 1)
    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"  axes: {', '.join(selected)}")
    print(f"  {uniq} unique configs (center deduped across axes) incl. "
          f"{len(args.heads)} head(s), x {len(args.seeds)} seeds"
          + (" x2 (replay + no-replay twin)" if args.with_noreplay else ""))
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
