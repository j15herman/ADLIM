"""
ABLATION: NEURON DEATH (structural turnover) in the DER++ x metaplasticity combo,
2026-07. Follow-up to gen_derpp_plast_ablate_sweep.py, which softened lambda/budget and
did NOT beat DER++ alone.

HYPOTHESIS: the brake center inherits `--plast_death weights_line` from the sweep13
winner, which was tuned WITHOUT replay. Death RECYCLES too-plastic neurons (reinit
incoming weights, respawn geometry, zero importance, reset the outgoing column). DER++
meanwhile replays old samples to keep the representation calibrated. These two fight:
death throws away exactly the units replay is re-stabilizing, and a recycled unit makes
every stored logit in the buffer partially stale. If that interaction is what caps the
combo below plain DER++, then making death RARER (or removing it) should recover the
gap. gen_derpp_plast_sweep.py flagged this from the start: "if the replay arms look
pathological, the first thing to try is --plast_death off on the brake center."

OFAT: every axis below varies ONLY its own death dial; everything else is held at the
base config (see BASE, overridable from the CLI), so each run is attributable to one
change. All axes push in the SAME direction -- LESS death.

  center         base config with death at its center setting -- the REFERENCE.
  death_off      --plast_death off. The clean "is death the problem at all" control.
                 If this alone closes the gap to DER++, the rest is just calibration.
  thresh         --plast_death_thresh   0.3 -> 0.2, 0.1, 0.05
                 With --plast_death_relative 1 (base) this is the TOP FRACTION of the
                 most-plastic tail that is ELIGIBLE: 0.3 = top-30%, 0.05 = top-5%.
                 Lower = a narrower eligible tail = fewer candidates.
  window         --plast_death_window   100 -> 200, 300, 400
                 CONSECUTIVE steps a neuron must stay above threshold before it may
                 die. Higher = only persistently-plastic units die.
  thresh_window  both together (0.2/200, 0.1/300, 0.05/400) -- the dials compose, and
                 either alone may be too weak to show the effect.
  rate_cap       --plast_death_rate_cap 0.05 -> 0.03, 0.02, 0.01
                 Max neurons that may die per step per layer, as a fraction of the
                 layer. Caps how fast turnover can run even when many are eligible.

TWO IMPLEMENTATION LIMITS worth knowing before you widen these ladders (both verified
against models/plasticity_mlp.py):

  * rate_cap floors at ONE neuron: _death_cap() returns max(1, int(cap * out_features)).
    The hidden layers are 100 units, so cap 0.01 -> 1 neuron/step and ANY smaller value
    is identical to 0.01. The ladder stops there on purpose; to go lower you would need
    --plast_death off, not a smaller cap.
  * window has a hard ceiling ~490. Death counters only accumulate AFTER
    --plast_death_warmup (500) and the run is 990 observe steps (198 steps/epoch x 1
    epoch x 5 tasks), so a neuron has ~490 post-warmup steps to build a streak -- and
    the streak RESETS whenever it drops below threshold. A window >= ~490 can never be
    met, i.e. it silently becomes death_off (already covered by that arm). 400 is
    deliberately near the edge: expect very few deaths there.

Read the `plast DEATH` / `plast GEO PROBE` lines in the .err logs for deaths-per-task
alongside the accuracy -- that is how you see whether an axis actually changed the
turnover rate or just moved a dial with no effect.

PROTOCOL and the DER++ hyperparameters are IDENTICAL to gen_derpp_plast_sweep.py
(buffer 500, alpha 1.0, beta 0.5, minibatch 10), so these runs drop into the same table
as the derpp / derpp_plast arms. Read mean +/- std over seeds TOGETHER -- single-seed
CIL here is a task-3 coin flip (std 3-6).

Self-contained (no sibling imports) -- copy just this file to the OSC checkout.
Requires models/plasticity_mlp.py with the --plast_derpp* flags AND no stale sibling in
models/ claiming NAME='plasticity_mlp' (a `plasticity_mlp_noderpp.py` once hijacked the
registry and silently ran the pre-derpp code -- verify with:
  python -c "from argparse import Namespace; from models import get_model_class; \
             import sys; c=get_model_class(Namespace(model='plasticity-mlp')); \
             print(sys.modules[c.__module__].__file__)")

Usage:
    python gen_derpp_plast_death_sweep.py --axes extremes      # MAX-DOSE: 5 x 10 = 50
    python gen_derpp_plast_death_sweep.py                     # all axes, 10 seeds
    python gen_derpp_plast_death_sweep.py --axes center death_off
    python gen_derpp_plast_death_sweep.py --points death=off rate_cap=0.01
    python gen_derpp_plast_death_sweep.py --base_lambda 80 \
        --base_budget_frac 0.2 --base_budget_frac_end 0.4     # pin a different base
    python gen_derpp_plast_death_sweep.py --with_noreplay     # matched brake-only twins
    python gen_derpp_plast_death_sweep.py --seeds 0 1 2 3 4   # half-cost scout pass
    sbatch --array=1-$(wc -l < derpp_plast_death_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_plast_death_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

# DER++ side, identical to the parent sweep so the arms are comparable.
DERPP = dict(buffer_size=500, minibatch=10, alpha=1.0, beta=0.5)

# BASE = the sweep13 winner brake center. The lambda/budget ablation did not beat DER++
# alone, so the brake dials are left AT the winner here and only DEATH is varied; if a
# different point turned out best, pin it with --base_lambda / --base_budget_frac /
# --base_budget_frac_end rather than editing this dict.
BASE = dict(
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

# Ablation axes: name -> [(label, {flag: value, ...}), ...], each overriding BASE.
AXES = {
    'death_off': [
        ('death=off', dict(plast_death='off')),
    ],
    'thresh': [
        ('thresh=0.2', dict(plast_death_thresh=0.2)),
        ('thresh=0.1', dict(plast_death_thresh=0.1)),
        ('thresh=0.05', dict(plast_death_thresh=0.05)),
    ],
    'window': [
        ('window=200', dict(plast_death_window=200)),
        ('window=300', dict(plast_death_window=300)),
        ('window=400', dict(plast_death_window=400)),
    ],
    'thresh_window': [
        ('thresh=0.2/window=200',
         dict(plast_death_thresh=0.2, plast_death_window=200)),
        ('thresh=0.1/window=300',
         dict(plast_death_thresh=0.1, plast_death_window=300)),
        ('thresh=0.05/window=400',
         dict(plast_death_thresh=0.05, plast_death_window=400)),
    ],
    # Stops at 0.01: _death_cap floors at max(1, int(cap*100)), so anything smaller
    # is the same 1-neuron-per-step cap.
    'rate_cap': [
        ('rate_cap=0.03', dict(plast_death_rate_cap=0.03)),
        ('rate_cap=0.02', dict(plast_death_rate_cap=0.02)),
        ('rate_cap=0.01', dict(plast_death_rate_cap=0.01)),
    ],
}

# Flat label -> override lookup across every axis (single source of truth for the
# named groups and for --points).
BY_LABEL = {label: over for entries in AXES.values() for label, over in entries}

# MAX-DOSE screening set: death off, plus the STRONGEST (least-death) rung of each
# knob. Use when you want the largest possible effect size per run and will only
# chase the dials that actually move the number. Reference = the `center` config,
# which is the same point as the derpp_plast_head arm of gen_derpp_plast_sweep.py,
# so it does not need re-running if you already have that arm.
EXTREMES = [
    'death=off',
    'thresh=0.05',
    'window=400',
    'thresh=0.05/window=400',
    'rate_cap=0.01',
]

DEFAULT_AXES = ['center'] + list(AXES)
ALL_AXES = DEFAULT_AXES + ['extremes']


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


def build_configs(selected, base, points=None):
    """(label, cfg) pairs for every requested ablation point.

    `points`, if given, restricts the result to those exact labels (applied after the
    axes are expanded) -- so you can cherry-pick individual rungs.
    """
    out = []
    for name in selected:
        if name == 'center':
            out.append(('center', dict(base)))
        elif name == 'extremes':
            for label in EXTREMES:
                cfg = dict(base)
                cfg.update(BY_LABEL[label])
                out.append((label, cfg))
        elif name in AXES:
            for label, over in AXES[name]:
                cfg = dict(base)
                cfg.update(over)
                out.append((label, cfg))

    if points:
        unknown = set(points) - set(BY_LABEL) - {'center'}
        if unknown:
            raise SystemExit(
                f"unknown --points label(s): {sorted(unknown)}\n"
                f"available: {sorted(BY_LABEL)} + 'center'")
        out = [(lbl, cfg) for lbl, cfg in out if lbl in set(points)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--axes', nargs='+', choices=ALL_AXES, default=None, metavar='AXIS',
                    help='Which ablation axes to emit (default: everything except '
                         '`extremes`, which is a subset of the others). `extremes` = '
                         'death off + the strongest rung of each knob (5 configs). '
                         'Choices: ' + ', '.join(ALL_AXES) + '.')
    ap.add_argument('--points', nargs='+', default=None, metavar='LABEL',
                    help='Cherry-pick individual rungs by label, e.g. --points '
                         'death=off thresh=0.05 rate_cap=0.01. Applied after --axes.')
    ap.add_argument('--heads', nargs='+', choices=['off', 'past_blocks'],
                    default=['past_blocks'], metavar='HEAD',
                    help='--plast_head_consolidate settings (default: past_blocks, the '
                         'sweep13 winner). Pass both to A/B the head freeze.')
    ap.add_argument('--with_noreplay', action='store_true',
                    help='Also emit each config WITHOUT DER++ (matched brake-only twin), '
                         'so the replay DELTA is measurable at every death setting. '
                         'Doubles the run count.')
    # Base-config overrides: keep "best config otherwise" without editing BASE.
    ap.add_argument('--base_lambda', type=float, default=None,
                    help='Override --plast_lambda in the base config.')
    ap.add_argument('--base_budget_frac', type=float, default=None,
                    help='Override --plast_budget_frac in the base config.')
    ap.add_argument('--base_budget_frac_end', type=float, default=None,
                    help='Override --plast_budget_frac_end in the base config.')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_plast_death_sweep.txt'))
    args = ap.parse_args()

    base = dict(BASE)
    for cli, flag in (('base_lambda', 'plast_lambda'),
                      ('base_budget_frac', 'plast_budget_frac'),
                      ('base_budget_frac_end', 'plast_budget_frac_end')):
        v = getattr(args, cli)
        if v is not None:
            base[flag] = v

    selected = args.axes or (ALL_AXES if args.points else DEFAULT_AXES)
    configs = build_configs(selected, base, points=args.points)
    if not configs:
        raise SystemExit("no configs selected -- check --axes / --points")

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

    uniq = len(lines) // max(len(args.seeds), 1)
    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"  axes: {', '.join(selected)}")
    print(f"  base: lambda={base['plast_lambda']} "
          f"budget={base['plast_budget_frac']}->{base['plast_budget_frac_end']}")
    print(f"  {uniq} unique configs incl. {len(args.heads)} head(s), "
          f"x {len(args.seeds)} seeds"
          + (" x2 (replay + no-replay twin)" if args.with_noreplay else ""))
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
