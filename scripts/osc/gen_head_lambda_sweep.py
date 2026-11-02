"""
LOWER HEAD-LAMBDA sweep on the top-1 plasticity_mlp config from latest_sweep4.out
(frac 0.875, dthr 0.5, layers=all, shared lambda 160). Uses the new per-head
override --plast_lambda_head to decouple the classifier's brake from fc1/fc2:
a SMALLER head lambda lets the readout keep training (stay plastic) while the
hidden layers stay stiffly consolidated at lambda 160.

Motivation: under layers=all the sweep4 CIL gain is a head-CALIBRATION effect
(freezing the head stops the last task owning the readout) but it cratered Task-IL
(94->86). Over-freezing the head may be what kills Task-IL; a lower head lambda
should freeze the head LESS / later, recovering Task-IL -- ideally without giving
back the Class-IL gain. Sweep the head brake well below the shared 160:

  plast_lambda_head: 5, 10, 20, 40, 80   (+ baseline: no override = head inherits 160)

Watch the classifier TURNOVER (frozen fraction, when) + Task-IL + oracle as
lambda_head falls. Bars (DER Seq-MNIST): oEWC 20.46/98.39, SGD 19.60/94.94;
sweep4 top-1 (head inherits 160) = 25.77/86.10, oracle 63.6.

NOTE: emits --plast_lambda_head (a NEW flag) -> SYNC models/plasticity_mlp.py to
the OSC checkout first. ≥5 seeds (variance ±5-7).

Usage:
    python gen_head_lambda_sweep.py [--seeds 0 1 2 3 4] [--out head_lambda.txt]
    sbatch --array=1-$(wc -l < head_lambda.txt)%24 \
        run_plasticity_mlp_sweep.sbatch head_lambda.txt
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

# Top-1 base: frac 0.875, dthr 0.5 (sweep4 rank-1, 25.77).
BASE = {'plast_budget_frac': 0.875, 'plast_death_thresh': 0.5}
HEAD_LAMBDAS = [5.0, 10.0, 20.0, 40.0, 80.0]


def build_configs():
    cfgs = [dict(BASE)]                                 # baseline: head inherits 160
    for lh in HEAD_LAMBDAS:
        cfgs.append({**BASE, 'plast_lambda_head': lh})
    return cfgs


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'head_lambda.txt'))
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
