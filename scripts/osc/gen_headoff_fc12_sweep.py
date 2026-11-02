"""
Head-OFF fc12 ablation sweep, 2026-07-06.

The head-freezing-OFF counterpart of the S-MNIST winner: identical hyperparameters
to the sweep13 best config (layers=fc12, lambda 160 accumulate, budget=neuron ramp,
conductance geo, weights_line death), but with `--plast_head_consolidate off` --
i.e. plasticity ONLY on fc1/fc2, no classifier-row freezing. Sweeps the two top
fc12 budget_frac_end values.

  layers=fc12  head_consolidate=off  x  budget_frac_end {0.8, 0.85}

= 2 configs x 10 seeds = 20 runs. (These mirror the head-off rows already in
sweep13 -- end0.8 -> ~20.1, end0.85 -> ~19.9 -- re-run here as a clean isolated
10-seed artifact.) All diagnostics ON. Report CIL + TIL + oracle together.

Self-contained (CENTER inlined verbatim); negative values emitted as --flag=value
so mammoth's check_multiple_defined_arg_during_string_parse never trips on -1.0.

Usage:
    python gen_headoff_fc12_sweep.py [--seeds 0 1 2 3 4 5 6 7 8 9] [--out headoff_fc12.txt]
    sbatch --array=1-$(wc -l < headoff_fc12.txt)%24 \
        run_plasticity_mlp_sweep.sbatch headoff_fc12.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# The sweep13 winner, copied verbatim, with the ONE change: head_consolidate=off
# (plasticity confined to fc1/fc2; the classifier head is plain SGD, never frozen).
CENTER = dict(
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
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.8,
    plast_head_consolidate='off',
)

BUDGET_FRAC_END = [0.8, 0.85]      # the two top fc12 past_blocks configs' values


def configs():
    return [dict(CENTER, plast_budget_frac_end=e) for e in BUDGET_FRAC_END]


def cfg_to_flags(cfg):
    # Emit negatives as --flag=value: a bare "-1.0" token trips mammoth's
    # check_multiple_defined_arg_during_string_parse (any '-'-prefixed token is
    # read as a flag name). The '=' form keeps the flag name as the token.
    out = []
    for k, v in cfg.items():
        s = str(v)
        out.append(f'--{k}={s}' if s.startswith('-') else f'--{k} {s}')
    return ' '.join(out)


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'headoff_fc12.txt'))
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    cfgs = configs()
    for seed in args.seeds:            # seed-outer, config-inner
        for cfg in cfgs:
            add(run_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({len(cfgs)} configs x {len(args.seeds)} seeds)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
