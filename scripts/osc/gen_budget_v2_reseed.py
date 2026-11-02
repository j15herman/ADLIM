"""
Reseed the TOP-10 plasticity_mlp configs from the budget_v2 sweep (the 5-seed
run, 2026-06-26) with 15 MORE seeds (5..19) to tighten mean±std and actually
discriminate the leaders -- they currently sit within ~1.4 CIL of each other
with overlapping CIs (sgd floor 19.57±0.12; best frac=0.1 20.97±0.91).

These are the overall-table ranks 1-9 and 11 (rank 10 is the MAS baseline,
skipped) = the 10 best plasticity-mlp runs. Each is expressed as a delta from
the v2 CENTER (imported verbatim, so every other flag matches the sweep exactly:
lam 80, nbar 32, frac 0.2, fc12, accumulate, conductance geo, weights_line death,
cosine LR, spread 1.0).

Combine with the original 5 seeds (0-4) for 20 seeds total per config.

Usage:
    python gen_budget_v2_reseed.py [--seeds 5 6 ... 19] [--out budget_v2_reseed.txt]
    sbatch --array=1-$(wc -l < budget_v2_reseed.txt)%24 \
        run_plasticity_mlp_sweep.sbatch budget_v2_reseed.txt
    # then aggregate the union of the old + new logs:
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Inlined so this generator is SELF-CONTAINED (no sibling-module imports) -- it
# runs wherever the file lands. PROTOCOL = Seq-MNIST sgd protocol; V2_CENTER = the
# frozen carved metaplast_branch_v2 baseline, verbatim from that generator.
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

V2_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='fc12',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='weights_line', plast_death_thresh=0.1, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
)

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

CENTER = dict(V2_CENTER)
CENTER['plast_budget'] = 'neuron'
CENTER['plast_budget_frac'] = 0.2          # == the v2 center; deltas below

# The top-10 (overall ranks 1-9, 11) as deltas from CENTER. Comment = final CIL
# mean±std at 5 seeds.
WINNERS = [
    {'plast_budget_frac': 0.9, 'plast_layers': 'all', 'plast_death_thresh': 0.5},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.875, 'plast_layers': 'all', 'plast_death_thresh': 0.5},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.85, 'plast_layers': 'all', 'plast_death_thresh': 0.5},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.825, 'plast_layers': 'all', 'plast_death_thresh': 0.5},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.9, 'plast_layers': 'all', 'plast_death_thresh': 0.4},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.875, 'plast_layers': 'all', 'plast_death_thresh': 0.4},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.85, 'plast_layers': 'all', 'plast_death_thresh': 0.4},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.825, 'plast_layers': 'all', 'plast_death_thresh': 0.4},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.9, 'plast_layers': 'all', 'plast_death_thresh': 0.3},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.875, 'plast_layers': 'all', 'plast_death_thresh': 0.3},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.85, 'plast_layers': 'all', 'plast_death_thresh': 0.3},          # 20.55±0.83  (#3)
    {'plast_budget_frac': 0.825, 'plast_layers': 'all', 'plast_death_thresh': 0.3},          # 20.55±0.83  (#3)
]


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(5, 20)))
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'budget_v2_reseed.txt'))
    args = ap.parse_args()

    seen, lines = set(), []
    for seed in args.seeds:                 # seed-outer, config-inner
        for delta in WINNERS:
            cfg = dict(CENTER); cfg.update(delta)
            line = run_line(cfg, seed)
            if line not in seen:
                seen.add(line); lines.append(line)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({len(WINNERS)} configs x {len(args.seeds)} seeds)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
