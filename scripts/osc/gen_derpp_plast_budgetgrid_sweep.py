"""
BUDGET 2-D GRID (start x end) on the DEATH-OFF baseline, 2026-07. Follow-up to
gen_derpp_plast_deathoff_sweep.py, which swept budget start and end as SEPARATE OFAT
axes. The two dials define one ramp together (plast_budget_frac -> plast_budget_frac_end),
so their effect is joint; OFAT can miss an interior optimum. This sweeps the full grid.

GRID (each cell sets BOTH ends of the per-neuron freezing-budget ramp):
    start (--plast_budget_frac)      in {0.05, 0.1, 0.2}
    end   (--plast_budget_frac_end)  in {0.8, 0.6, 0.4}
  = 9 cells. All have end > start, i.e. the budget GROWS over the run (plastic early,
  more freezing allowed late) -- from a much scarcer start than the death-off center's
  0.4 -> 0.8. Read as a heatmap over (start, end).

Plus `center` (the death-off winner, budget 0.4 -> 0.8, CIL 91.63) as the in-batch
reference so this batch is self-contained (no cross-submission drift). Drop it with
--no_center; change the grid with --starts / --ends.

Everything else is the death-off base: --plast_death off, lambda 160, p_max 20,
stiffness 2, past_blocks head, DER++ buffer 500 / alpha 1.0 / beta 0.5. Strict: only the
two budget flags move across the grid. Read mean +/- std over seeds TOGETHER -- single-
seed CIL is a task-3 coin flip (std 3-6).

Self-contained (no sibling imports) -- copy just this file to the OSC checkout. Requires
models/plasticity_mlp.py with the --plast_derpp* flags AND no stale sibling in models/
claiming NAME='plasticity_mlp' (verify with:
  python -c "from argparse import Namespace; from models import get_model_class; \
             import sys; c=get_model_class(Namespace(model='plasticity-mlp')); \
             print(sys.modules[c.__module__].__file__)")

Usage:
    python gen_derpp_plast_budgetgrid_sweep.py                 # 9 cells + center, 10 seeds
    python gen_derpp_plast_budgetgrid_sweep.py --no_center
    python gen_derpp_plast_budgetgrid_sweep.py --starts 0.05 0.1 0.2 0.3 --ends 0.9 0.7 0.5
    python gen_derpp_plast_budgetgrid_sweep.py --base_lambda 40   # re-center for round 2
    sbatch --array=1-$(wc -l < derpp_plast_budgetgrid_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_plast_budgetgrid_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

DERPP = dict(buffer_size=500, minibatch=10, alpha=1.0, beta=0.5)

# BASE = the winning DEATH-OFF config (sweep13 brake center + --plast_death off). The
# two budget flags are overwritten per grid cell; everything else is held.
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


def cfg_to_flags(cfg):
    # Emit NEGATIVE values as `--flag=value` so a bare "-1.0" is never parsed as a flag
    # token (mammoth treats any '-'-prefixed token as a flag name).
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


def build_configs(base, starts, ends, with_center):
    """(label, cfg) pairs: optional center + every (start, end) grid cell."""
    out = []
    if with_center:
        out.append(('center', dict(base)))
    for s in starts:
        for e in ends:
            cfg = dict(base)
            cfg['plast_budget_frac'] = s
            cfg['plast_budget_frac_end'] = e
            out.append((f'budget={s}->{e}', cfg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--starts', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                    help='Ramp START values (--plast_budget_frac). Default 0.05 0.1 0.2.')
    ap.add_argument('--ends', type=float, nargs='+', default=[0.8, 0.6, 0.4],
                    help='Ramp END values (--plast_budget_frac_end). Default 0.8 0.6 0.4.')
    ap.add_argument('--no_center', action='store_true',
                    help='Omit the death-off center (budget 0.4->0.8) reference point.')
    ap.add_argument('--heads', nargs='+', choices=['off', 'past_blocks'],
                    default=['past_blocks'], metavar='HEAD',
                    help='--plast_head_consolidate settings (default: past_blocks).')
    ap.add_argument('--with_noreplay', action='store_true',
                    help='Also emit each cell WITHOUT DER++ (matched brake-only twin). '
                         'Doubles the run count.')
    # Re-center the non-budget base for a follow-up round without editing BASE.
    ap.add_argument('--base_lambda', type=float, default=None)
    ap.add_argument('--base_p_max', type=float, default=None)
    ap.add_argument('--base_stiffness', type=float, default=None)
    ap.add_argument('--base_death', type=str, default=None,
                    choices=['off', 'weights_line', 'line_only', 'synapse'],
                    help='Override base --plast_death (default: off).')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_plast_budgetgrid_sweep.txt'))
    args = ap.parse_args()

    base = dict(BASE)
    for cli, flag in (('base_lambda', 'plast_lambda'),
                      ('base_p_max', 'plast_p_max'),
                      ('base_stiffness', 'plast_consol_stiffness'),
                      ('base_death', 'plast_death')):
        v = getattr(args, cli)
        if v is not None:
            base[flag] = v

    # Sanity: warn (do not block) on any cell where the ramp does not grow.
    for s in args.starts:
        for e in args.ends:
            if e < s:
                print(f"  note: cell {s}->{e} SHRINKS the budget (end < start).")

    configs = build_configs(base, args.starts, args.ends, not args.no_center)

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
    print(f"  grid: starts {args.starts} x ends {args.ends} = "
          f"{len(args.starts) * len(args.ends)} cells"
          + ("" if args.no_center else " + center"))
    print(f"  base: death={base['plast_death']} lambda={base['plast_lambda']} "
          f"p_max={base['plast_p_max']} stiffness={base['plast_consol_stiffness']}")
    print(f"  {uniq} unique configs incl. {len(args.heads)} head(s), "
          f"x {len(args.seeds)} seeds"
          + (" x2 (replay + no-replay twin)" if args.with_noreplay else ""))
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
