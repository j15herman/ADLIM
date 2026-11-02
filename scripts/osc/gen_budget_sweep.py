"""
Per-neuron consolidation-BUDGET sweep for plasticity_mlp on Seq-MNIST
(2026-06-25). Sibling of gen_plasticity_mlp_sweep.py / gen_metaplast_branch_v2.

Motivation. The turnover/freeze-timing probe showed consolidation is heavily
FRONT-LOADED: at the simple accumulate baseline (lambda 20, floor 1.0) fc1
freezes ~30% of its weights almost entirely during task 0 (when~=0.01), leaving
little plasticity for later tasks. The new --plast_budget neuron mechanism caps
each postsynaptic neuron's total freezing mass at --plast_budget_frac * fan_in
(synapses COMPETE for a finite, per-step pool), so no single task can lock a
whole neuron. This sweep finds where budget_frac x lambda actually lifts
retention vs the unbudgeted baseline.

Built on the SIMPLE proven center (no geometry/death/spread/stiffness) to ISOLATE
the budget's effect. Every run logs the turnover probe (`plast TURNOVER task T`)
plus the usual NCM oracle / PROBE, so you can read across the array whether the
budget spreads freezing across tasks (lower task-0 `new`, higher later `new`).

Baselines (sgd, plast-OFF sanity, SI, MAS @ e_lambda {0.1,1,10}) are emitted by
DEFAULT here (the MAS hang at e_lambda>=1 is fixed in models/mas.py, so the
higher-lambda MAS bars now complete). Pass --no_baselines to skip them.

Usage:
    python gen_budget_sweep.py [--seeds 0 1 2 3 4] [--out budget_sweep.txt]
    sbatch --array=1-$(wc -l < budget_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch budget_sweep.txt

Read the diagnostics from the slurm .out files:
    Select-String *.out -Pattern 'Accuracy for 5 task'   # CIL/TIL + raw vector
    Select-String *.out -Pattern 'plast TURNOVER task'    # per-task freeze turnover
    Select-String *.out -Pattern 'plast NCM ORACLE task 4'
"""

import argparse
import os

from gen_plasticity_mlp_sweep import PROTOCOL, baseline_lines

HERE = os.path.dirname(os.path.abspath(__file__))

# Turnover probe ON for every run (log-only: no --plast_turnover_path, so no
# per-run .pt dumps cluttering the array; the TURNOVER log lines are enough to
# read freeze-timing across configs. Dump+plot a single chosen config separately).
PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 '
            '--plast_ncm_oracle 1 --plast_probe 1 --plast_turnover_probe 1')

# Simple proven center + the per-neuron budget at a middle anchor frac.
CENTER = dict(
    plast_signal='act_grad',
    plast_consistency='mean',
    plast_beta=0.99,
    plast_lambda=20.0,
    plast_normalize=1,
    plast_p_min=0.0,
    plast_p_max=20.0,
    plast_layers='fc12',
    plast_importance_mode='accumulate',
    plast_accum_floor=1.0,           # monotonic (proven center)
    plast_accum_center=1,
    plast_warmup_epochs=0.0,
    # --- the new mechanism ---
    plast_budget='neuron',
    plast_budget_frac=0.2,           # anchor; swept below
)

# OFAT axes -- the headline is budget_frac x lambda.
AXES = {
    # HEADLINE: per-neuron freezing budget. 1.0 = ~loose (rarely binds);
    # 0.05 = very scarce (reserves most plasticity). Find the sweet spot.
    'plast_budget_frac': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    # brake strength interacts: higher lambda freezes harder, the budget caps how
    # much of it lands at once -- so the budget should let high lambda be safe.
    'plast_lambda': [10.0, 20.0, 40.0, 80.0],
    # recency vs persistent under a budget.
    'plast_importance_mode': ['ema', 'accumulate'],
}

# Coupled configs: the A/B control + the interesting corners.
EXTRA_CONFIGS = [
    # THE CONTROL: budget OFF at the otherwise-identical center (== prior
    # accumulate baseline). The direct A/B for "does the budget help".
    {'plast_budget': 'none'},
    # Tight budget + HIGH lambda: strong-but-rationed freezing (the hypothesis --
    # high lambda picks sharp winners, the budget stops task 0 hoarding them).
    {'plast_budget_frac': 0.1, 'plast_lambda': 80.0},
    {'plast_budget_frac': 0.05, 'plast_lambda': 80.0},
    # Loose budget + high lambda: should ~= unbudgeted high lambda (over-freeze).
    {'plast_budget_frac': 1.0, 'plast_lambda': 80.0},
    # Tight budget + ema (recency): does rationing recency-freezing help retention?
    {'plast_budget_frac': 0.1, 'plast_importance_mode': 'ema'},
    # Budget + erodable floor (the other retention knob) to see if they stack.
    {'plast_budget_frac': 0.1, 'plast_accum_floor': -1.0},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg: dict, seed: int) -> str:
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'budget_sweep.txt'))
    ap.add_argument('--no_baselines', action='store_true',
                    help='skip the sgd/SI/MAS baseline rows (emitted by default).')
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line)
            lines.append(line)

    # seed-outer, config-inner (so reseeding specific configs is line arithmetic).
    for seed in args.seeds:
        if not args.no_baselines:
            for bl in baseline_lines(seed):
                add(bl)
        add(run_line(dict(CENTER), seed))                  # center anchor
        for axis, values in AXES.items():                  # OFAT axes
            for v in values:
                cfg = dict(CENTER)
                cfg[axis] = v
                add(run_line(cfg, seed))
        for extra in EXTRA_CONFIGS:                         # coupled configs
            cfg = dict(CENTER)
            cfg.update(extra)
            add(run_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    n_per_seed = len(lines) // max(len(args.seeds), 1)
    print(f"wrote {len(lines)} runs to {args.out} "
          f"(~{n_per_seed} configs/seed x {len(args.seeds)} seeds)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
