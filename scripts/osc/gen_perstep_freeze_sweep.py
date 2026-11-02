"""
Two requested ablations on the v2 (carved metaplast_branch_v2 + budget) center,
2026-06-26 (revised after the ema/accumulate correction):

  (A) TRUE PER-STEP importance signal -- remove the EMA that smooths the
      importance, WHILE STAYING IN accumulate mode. Rationale (user): `ema` mode
      is NOT the local/per-step one -- its p=1/(1+lambda*Omega/Omega.mean())
      normalizes by the per-layer mean, i.e. GLOBAL info outside the weight's
      instantaneous state, and it underperforms (sweep-1: ema~0 vs accumulate
      ~20% retention). `accumulate` is the keeper: A += (Omega - mean) integrates
      the weight's DEVIATION from the running steady-state baseline ("a
      perturbation to steady state"). The only EMA to drop is the signal-moment
      EMA --plast_beta: at beta=0, Omega each step is the CURRENT batch's
      activation-gradient moments (no smoothing), and that per-step signal feeds
      the accumulator. So the per-step axis is a beta ladder {0, .5, .9, .99} in
      accumulate mode (beta=0 = true per-step; 0.99 = the v2 center reference),
      plus beta=0 crossed with budget_frac and lambda. Read churn off
      `plast TURNOVER` thaw/when.

  (B) INCREASE FREEZING via a bigger per-neuron budget. budget_frac caps each
      neuron's freezing mass; raising it lets more freeze. ~frac 1.0 already
      reaches the unbudgeted level (~57% of fc1 froze in the 5-seed sweep), so
      frac {2,5} just fully disable the cap; the two budget x high-lambda crosses
      push freezing deeper past that saturation. On the accumulate center.

Built on the v2 CENTER (imported verbatim; importance_mode=accumulate). Baselines
OFF by default (--with_baselines to re-emit). Turnover/probe/oracle/geo all ON.

Usage:
    python gen_perstep_freeze_sweep.py [--seeds 0 1 2 3 4] [--out perstep_freeze_sweep.txt]
    sbatch --array=1-$(wc -l < perstep_freeze_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch perstep_freeze_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Inlined so this generator is SELF-CONTAINED -- no sibling-module imports, so it
# runs wherever the file lands regardless of what else is in scripts/osc. PROTOCOL
# is the Seq-MNIST sgd protocol; V2_CENTER is the frozen carved
# metaplast_branch_v2 baseline, copied verbatim from gen_metaplast_branch_v2_sweep.py.
# (Baselines, only used with --with_baselines, are imported lazily in main.)
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

V2_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=80.0, plast_lambda_end=-1.0, plast_normalize=1,
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

# v2 center: importance_mode=accumulate, beta=0.99, frac 0.2, lam 80, nbar 32.
CENTER = dict(V2_CENTER)
CENTER['plast_budget'] = 'neuron'
CENTER['plast_budget_frac'] = 0.2

# (A) per-step axis = drop the signal-EMA (beta) IN ACCUMULATE MODE.
PERSTEP_CONFIGS = [
    # beta ladder at the center (beta 0 = true per-step ... 0.99 = v2 center ref).
    {'plast_beta': 0.0},
    {'plast_beta': 0.5},
    {'plast_beta': 0.9},
    {'plast_beta': 0.99},
    # true per-step crossed with a tighter / looser budget...
    {'plast_beta': 0.0, 'plast_budget_frac': 0.1},
    {'plast_beta': 0.0, 'plast_budget_frac': 1.0},
    # ...and a lower / higher brake.
    {'plast_beta': 0.0, 'plast_lambda': 20.0},
    {'plast_beta': 0.0, 'plast_lambda': 160.0},
]

# (B) increase freezing via a bigger per-neuron budget (accumulate, beta 0.99).
FREEZE_UP_CONFIGS = [
    {'plast_budget_frac': 0.5},
    {'plast_budget_frac': 1.0},
    {'plast_budget_frac': 2.0},          # >=1 fully disables the cap
    {'plast_budget_frac': 5.0},
    {'plast_budget_frac': 2.0, 'plast_lambda': 160.0},
    {'plast_budget_frac': 5.0, 'plast_lambda': 160.0},
]


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'perstep_freeze_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true')
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    baseline_lines = None
    if args.with_baselines:                 # lazy: only this path needs the sibling
        from gen_plasticity_mlp_sweep import baseline_lines

    for seed in args.seeds:                 # seed-outer, config-inner
        if args.with_baselines:
            for bl in baseline_lines(seed):
                add(bl)
        for delta in PERSTEP_CONFIGS + FREEZE_UP_CONFIGS:
            cfg = dict(CENTER); cfg.update(delta)
            add(run_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    n_cfg = len(PERSTEP_CONFIGS) + len(FREEZE_UP_CONFIGS)
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({n_cfg} configs x {len(args.seeds)} seeds"
          f"{' + baselines' if args.with_baselines else ''})")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
