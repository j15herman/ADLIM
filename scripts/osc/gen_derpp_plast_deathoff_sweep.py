"""
DEATH-OFF baseline: OFAT sweep of the remaining plasticity knobs in the DER++ x
metaplasticity combo, 2026-07. Follow-up to gen_derpp_plast_death_sweep.py.

RESULT SO FAR: turning neuron death OFF is the single best change found -- death=off
gives CIL 91.63 +/- 0.43 vs 90.20 for the strongest death-on rung (dthr 0.05) and 87
for the winner center's death. Replay re-stabilizes old samples; recycling neurons
throws away exactly the units replay is keeping calibrated (and staleness every stored
logit). So the NEW BASELINE fixes --plast_death off and we re-open the OTHER
consolidation dials, which were all tuned WITHOUT replay AND with death ON -- they may
now sit at the wrong point.

DIRECTION: the story is "less consolidation helps once replay carries the load," so each
knob is probed UP A BIT (one modest step above the center, in case the optimum is just
past it) and DOWN A LOT (a wide ladder below, since that is where the hypothesis points).
Strict OFAT: every run changes ONE dial off the death-off base; everything else held.

  lambda        --plast_lambda            200 | 80, 40, 20     (center 160)
                Brake strength. p = 1/(1 + lambda*Omega_hat); lower = weaker brake.
  budget_start  --plast_budget_frac       0.5 | 0.2, 0.1, 0.05 (center 0.4; end=0.8)
                Per-neuron freezing budget as a fraction of fan-in, at the ramp START.
  budget_end    --plast_budget_frac_end   0.9 | 0.6, 0.4, 0.2  (center 0.8; start=0.4)
                Where the budget ramp LANDS. 0.4 = constant; 0.2 = budget SHRINKS over
                the run (scarcer freezing late).
  p_max         --plast_p_max             30  | 10, 5, 2       (center 20)
                Upper clamp on per-weight plasticity -- the re-plasticize BOOST ceiling
                for accumulate mode. VERIFIED BINDING: the death=off log shows p[.9]=20
                and p_plas~19.9, i.e. weights are pinned at this ceiling, so lowering it
                actually changes the dynamics (it is not a slack clamp).
  stiffness     --plast_consol_stiffness  3   | 1, 0.5, 0      (center 2)
                Consolidation-dependent (Benna-Fusi) accumulator gain. 0 = off (gain 1
                everywhere); higher = consolidated synapses get stickier faster.

The center point (death off, all knobs at the winner value + DER++) is emitted ONCE as
`center` -- the reference every axis is read against, and the SAME config as the winning
death=off run, so it re-measures your baseline in-batch (no cross-submission drift).

PROTOCOL and DER++ hyperparameters IDENTICAL to gen_derpp_plast_sweep.py (buffer 500,
alpha 1.0, beta 0.5, minibatch 10). Read mean +/- std over seeds TOGETHER -- single-seed
CIL here is a task-3 coin flip (std 3-6).

Self-contained (no sibling imports) -- copy just this file to the OSC checkout. Requires
models/plasticity_mlp.py with the --plast_derpp* flags AND no stale sibling in models/
claiming NAME='plasticity_mlp' (verify with:
  python -c "from argparse import Namespace; from models import get_model_class; \
             import sys; c=get_model_class(Namespace(model='plasticity-mlp')); \
             print(sys.modules[c.__module__].__file__)")

Usage:
    python gen_derpp_plast_deathoff_sweep.py                  # all axes, 10 seeds = 210
    python gen_derpp_plast_deathoff_sweep.py --axes lambda p_max
    python gen_derpp_plast_deathoff_sweep.py --points center lambda=40 stiffness=0
    python gen_derpp_plast_deathoff_sweep.py --base_lambda 80   # re-center for round 2
    python gen_derpp_plast_deathoff_sweep.py --seeds 0 1 2 3 4  # half-cost scout pass
    sbatch --array=1-$(wc -l < derpp_plast_deathoff_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_plast_deathoff_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

DERPP = dict(buffer_size=500, minibatch=10, alpha=1.0, beta=0.5)

# BASE = the winning DEATH-OFF config (the sweep13 brake center with --plast_death off).
# Death params are kept (inert under 'off') so an emitted line matches the proven
# death=off run byte-for-byte except the single swept knob.
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
    plast_death='off', plast_death_thresh=0.3, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.80,
)

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# OFAT axes: name -> [(label, {flag: value}), ...]. First entry of each = "up a bit",
# the rest = "down a lot". None equals the center value, so nothing dedupes away.
AXES = {
    'lambda': [
        ('lambda=200', dict(plast_lambda=200.0)),
        ('lambda=80', dict(plast_lambda=80.0)),
        ('lambda=40', dict(plast_lambda=40.0)),
        ('lambda=20', dict(plast_lambda=20.0)),
    ],
    'budget_start': [
        ('budget_frac=0.5', dict(plast_budget_frac=0.5)),
        ('budget_frac=0.2', dict(plast_budget_frac=0.2)),
        ('budget_frac=0.1', dict(plast_budget_frac=0.1)),
        ('budget_frac=0.05', dict(plast_budget_frac=0.05)),
    ],
    'budget_end': [
        ('budget_end=0.9', dict(plast_budget_frac_end=0.9)),
        ('budget_end=0.6', dict(plast_budget_frac_end=0.6)),
        ('budget_end=0.4', dict(plast_budget_frac_end=0.4)),
        ('budget_end=0.2', dict(plast_budget_frac_end=0.2)),
    ],
    'p_max': [
        ('p_max=30', dict(plast_p_max=30.0)),
        ('p_max=10', dict(plast_p_max=10.0)),
        ('p_max=5', dict(plast_p_max=5.0)),
        ('p_max=2', dict(plast_p_max=2.0)),
    ],
    'stiffness': [
        ('stiffness=3', dict(plast_consol_stiffness=3.0)),
        ('stiffness=1', dict(plast_consol_stiffness=1.0)),
        ('stiffness=0.5', dict(plast_consol_stiffness=0.5)),
        ('stiffness=0', dict(plast_consol_stiffness=0.0)),
    ],
}

BY_LABEL = {label: over for entries in AXES.values() for label, over in entries}
ALL_AXES = ['center'] + list(AXES)


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
    """(label, cfg) pairs for every requested point; `points` restricts to exact labels."""
    out = []
    for name in selected:
        if name == 'center':
            out.append(('center', dict(base)))
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
                    help='Which axes to emit (default: all). Choices: '
                         + ', '.join(ALL_AXES) + '.')
    ap.add_argument('--points', nargs='+', default=None, metavar='LABEL',
                    help='Cherry-pick individual rungs by label, e.g. --points center '
                         'lambda=40 p_max=5. Applied after --axes.')
    ap.add_argument('--heads', nargs='+', choices=['off', 'past_blocks'],
                    default=['past_blocks'], metavar='HEAD',
                    help='--plast_head_consolidate settings (default: past_blocks).')
    ap.add_argument('--with_noreplay', action='store_true',
                    help='Also emit each config WITHOUT DER++ (matched brake-only twin). '
                         'Doubles the run count.')
    # Re-center the base for a follow-up round without editing BASE.
    ap.add_argument('--base_lambda', type=float, default=None)
    ap.add_argument('--base_budget_frac', type=float, default=None)
    ap.add_argument('--base_budget_frac_end', type=float, default=None)
    ap.add_argument('--base_p_max', type=float, default=None)
    ap.add_argument('--base_stiffness', type=float, default=None)
    ap.add_argument('--base_death', type=str, default=None,
                    choices=['off', 'weights_line', 'line_only', 'synapse'],
                    help='Override the base --plast_death (default: off).')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_plast_deathoff_sweep.txt'))
    args = ap.parse_args()

    base = dict(BASE)
    for cli, flag in (('base_lambda', 'plast_lambda'),
                      ('base_budget_frac', 'plast_budget_frac'),
                      ('base_budget_frac_end', 'plast_budget_frac_end'),
                      ('base_p_max', 'plast_p_max'),
                      ('base_stiffness', 'plast_consol_stiffness'),
                      ('base_death', 'plast_death')):
        v = getattr(args, cli)
        if v is not None:
            base[flag] = v

    selected = args.axes or ALL_AXES
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
    print(f"  base: death={base['plast_death']} lambda={base['plast_lambda']} "
          f"budget={base['plast_budget_frac']}->{base['plast_budget_frac_end']} "
          f"p_max={base['plast_p_max']} stiffness={base['plast_consol_stiffness']}")
    print(f"  {uniq} unique configs incl. {len(args.heads)} head(s), "
          f"x {len(args.seeds)} seeds"
          + (" x2 (replay + no-replay twin)" if args.with_noreplay else ""))
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
