"""
HEAD-LAMBDA RAMP-UP sweep on the top-1 plasticity_mlp config from latest_sweep4.out
(frac 0.875, dthr 0.5, layers=all). Uses the new per-head overrides
--plast_lambda_head (start) + --plast_lambda_end_head (end) so the classifier's
brake RAMPS UP over the run while fc1/fc2 stay at the constant shared lambda 160.

This is handoff rec #2 made precise: "train the head first, freeze it late."
Early in the run the head brake is small -> the readout trains freely (good for
Task-IL / clean features); by the end it ramps to a strong brake -> the head
consolidates (the calibration effect that lifts Class-IL). Goal: keep BOTH
Task-IL (which the constant strong head-freeze in sweep4 cratered to 86) and the
Class-IL bump, instead of trading one for the other.

  start (lambda_head):  5, 10, 20
  end   (lambda_end_head): 80, 160     -> 6 ramps + baseline (head inherits 160) = 7

Watch the classifier TURNOVER `when` shift LATE (fresh freezes near end-of-run)
and Task-IL recover vs sweep4 top-1 (25.77/86.10, oracle 63.6). Bars (DER
Seq-MNIST): oEWC 20.46/98.39, SGD 19.60/94.94.

NOTE: emits --plast_lambda_head / --plast_lambda_end_head (NEW flags) -> SYNC
models/plasticity_mlp.py to the OSC checkout first. ≥5 seeds (variance ±5-7).

Usage:
    python gen_head_lambda_ramp_sweep.py [--seeds 0 1 2 3 4] [--out head_lambda_ramp.txt]
    sbatch --array=1-$(wc -l < head_lambda_ramp.txt)%24 \
        run_plasticity_mlp_sweep.sbatch head_lambda_ramp.txt
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
STARTS = [5.0, 10.0, 20.0]      # early head brake (head trains freely)
ENDS = [80.0, 160.0]            # late head brake (head consolidates)


def build_configs():
    cfgs = [dict(BASE)]                                 # baseline: head inherits 160
    for end in ENDS:
        for start in STARTS:
            cfgs.append({**BASE,
                         'plast_lambda_head': start,
                         'plast_lambda_end_head': end})
    return cfgs


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'head_lambda_ramp.txt'))
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
