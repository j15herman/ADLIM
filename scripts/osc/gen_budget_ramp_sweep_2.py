"""
Budget RAMP-UP sweep on the top-3 plasticity_mlp configs from latest_sweep4.out.
Tests the new --plast_budget_frac_end (per-neuron budget ramps start->end over the
run, like --plast_lambda_end): GROW the freezing budget over time so the network
stays plastic early (reserve capacity, learn the task) and consolidates more late.
Hypothesis: a low-start / high-end ramp keeps the early-task representation intact
(helping Task-IL / oracle, which sweep4's high constant budget hurt) while still
ending at the high frozen mass that drove the CIL bump.

For each top-3 config the budget ramps from `start` UP to that config's frac
(the sweep4 value = the ramp END), keeping its death setting. Plus the 3 constant
(no-ramp) leaders as in-sweep references.

  ends  : 0.875 (top-1, dthr0.5) | 0.9 (top-2, dthr0.3) | 0.85 (top-3, dthr0.4)
  starts: 0.1, 0.3, 0.5   -> 9 ramp configs + 3 constant references = 12

Bars (DER Seq-MNIST): oEWC 20.46/98.39, SGD 19.60/94.94. sweep4 constant top-3 =
24.7-25.8 CIL @ 86-90 TIL, oracle ~64. Win = hold CIL while lifting TIL/oracle.

NOTE: emits --plast_budget_frac_end (a NEW flag) -> SYNC models/plasticity_mlp.py
to the OSC checkout before running. ≥5 seeds (variance ±5-7).

Usage:
    python gen_budget_ramp_sweep_2.py --seeds 0 1 2 3 4 5 6 7 8 9 --out budget_ramp_2.txt
    sbatch --array=1-$(wc -l < budget_ramp_top3.txt)%24 \
        run_plasticity_mlp_sweep.sbatch budget_ramp_top3.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

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

# (end budget = sweep4 frac, death_thresh) for each leader.
TOP3 = [
    (0.8,  0.2),    # rank 1 (25.77)
    (0.8,  0.25),    # rank 2 
    (0.8,  0.3),    # rank 2 (24.90)
    (0.85,  0.3),    # rank 2 (24.90)
]
STARTS = [0.4]


def build_configs():
    cfgs = []
    for end, dthr in TOP3:
        cfgs.append({'plast_budget_frac': end,        # constant reference (no ramp)
                     'plast_death_thresh': dthr})
        for start in STARTS:
            cfgs.append({'plast_budget_frac': start,  # ramp start -> end over run
                         'plast_budget_frac_end': end,
                         'plast_death_thresh': dthr})
    return cfgs


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'budget_ramp_top3.txt'))
    args = ap.parse_args()

    configs = build_configs()
    seen, lines = set(), []
    for seed in args.seeds:                 # seed-outer, config-inner
        for delta in configs:
            cfg = dict(SWEEP4_CENTER); cfg.update(delta)
            line = run_line(cfg, seed)
            if line not in seen:
                seen.add(line); lines.append(line)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({len(configs)} configs x {len(args.seeds)} seeds)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()