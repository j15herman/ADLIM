"""
Death-OFF isolation for the top-3 plasticity_mlp configs from latest_sweep4.out
(layers=all + per-neuron budget + neuron death, 11-seed, 2026-06-26). sweep4 hit
Class-IL 24-26 but Task-IL CRASHED to ~86 (below the SGD 94.94 floor) and the NCM
oracle DROPPED to ~64 -- the death machinery (450-1200 neurons recycled/task) is
suspected of damaging the representation while the layers=all head-freeze does the
CIL work. This sweep turns DEATH OFF on the three leaders to test the split:

  - if CIL holds (~24-25) with death off -> death was NOT the driver; we keep the
    CIL gain AND recover Task-IL / oracle (the win).
  - if CIL falls toward ~21 -> death was load-bearing for the CIL bump (costly).

Report CIL *and* Task-IL *and* oracle together (this is a Pareto question).

Configs (deltas from the sweep4 CENTER = v2 center + layers=all + lambda 160 +
budget=neuron; death_thresh inert when death=off so left at the center 0.1):
  1. frac 0.875, death off          (was rank-1 dthr0.5, 25.77)
  2. frac 0.9,   death off          (was rank-2 dthr0.3, 24.90)
  3. frac 0.85,  death off          (was rank-3 dthr0.4, 24.72)
  4. frac 0.9,   death ON (dthr 0.1) -- mild-death reference at frac 0.9; pairs
     with #2 for the clean death on/off contrast at the same budget.

Bars (DER Seq-MNIST, identical protocol): oEWC 20.46/98.39, SGD 19.60/94.94.
sweep4 frac0.9 layers=all death-on @ dthr0.1 was ~20.6-21.3 (logs2_5/logs3, 5sd).

Uses only EXISTING flags (--plast_death off) -> runs on the current OSC checkout
without syncing the model. METHOD: ±5-7 CIL variance here; bump --seeds to 0..9
for the decisive death-off vs death-on read.

Usage:
    python gen_deathoff_top3_sweep.py [--seeds 0 1 2 3 4] [--out deathoff_top3.txt]
    sbatch --array=1-$(wc -l < deathoff_top3.txt)%24 \
        run_plasticity_mlp_sweep.sbatch deathoff_top3.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Self-contained (no sibling imports). PROTOCOL = Seq-MNIST sgd protocol;
# SWEEP4_CENTER = the latest_sweep4.out center verbatim (v2 center + layers=all +
# lambda 160 + budget=neuron). Per-config budget_frac / death_thresh are deltas.
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

SWEEP4_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='all',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='weights_line', plast_death_thresh=0.1, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_budget='neuron', plast_budget_frac=0.2,
)

CONFIGS = [
    {'plast_budget_frac': 0.875, 'plast_death': 'off'},   # top-1 frac, death off
    {'plast_budget_frac': 0.9,   'plast_death': 'off'},   # top-2 frac, death off
    {'plast_budget_frac': 0.85,  'plast_death': 'off'},   # top-3 frac, death off
    {'plast_budget_frac': 0.9},                           # frac 0.9, death ON dthr0.1
]


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'deathoff_top3.txt'))
    args = ap.parse_args()

    seen, lines = set(), []
    for seed in args.seeds:                 # seed-outer, config-inner
        for delta in CONFIGS:
            cfg = dict(SWEEP4_CENTER); cfg.update(delta)
            line = run_line(cfg, seed)
            if line not in seen:
                seen.add(line); lines.append(line)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({len(CONFIGS)} configs x {len(args.seeds)} seeds)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
