"""
Per-neuron consolidation-BUDGET sweep on the MOST RECENT (metaplast_branch_v2)
baseline for plasticity_mlp on Seq-MNIST (2026-06-25).

Unlike gen_budget_sweep.py (which isolates the budget on the SIMPLE center), this
builds the budget on top of the carved-geometry v2 CENTER -- imported verbatim
from gen_metaplast_branch_v2_sweep.py, so it tracks your most recent runs:
lambda 80, consol_stiffness 2.0, stiff_max 1000, conductance geometry
(n_barriers 32, sigma 0.02, height 8, width 0.2), weight-line death, accumulate,
cosine LR, spread 1.0. The only thing added is the new --plast_budget neuron
mechanism (cap each neuron's total freezing mass at --plast_budget_frac*fan_in so
synapses compete for a finite per-step pool, recomputed every step).

Headline = budget_frac x lambda on that carved baseline. Every run logs the new
turnover/freeze-timing probe (`plast TURNOVER task T`) alongside the v2
diagnostics (NCM oracle, PROBE, GEO PROBE), so you can read whether the budget
spreads freezing across tasks (lower task-0 `new`, higher later `new`) and
whether that lifts retention vs the budget-OFF control (== the plain v2 center).

Baselines (sgd, plast-OFF sanity, SI, MAS @ e_lambda {0.1,1,10}) are emitted by
DEFAULT (the MAS hang at e_lambda>=1 is fixed in models/mas.py). --no_baselines
to skip.

Usage:
    python gen_budget_v2_sweep.py [--seeds 0 1 2 3 4] [--out budget_v2_sweep.txt]
    sbatch --array=1-$(wc -l < budget_v2_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch budget_v2_sweep.txt

Read the diagnostics from the slurm .out files:
    Select-String *.out -Pattern 'Accuracy for 5 task'    # CIL/TIL + raw vector
    Select-String *.out -Pattern 'plast TURNOVER task'     # per-task freeze turnover
    Select-String *.out -Pattern 'plast NCM ORACLE task 4'
    Select-String *.out -Pattern 'plast GEO PROBE task 4'  # eff_comp / deaths
"""

import argparse
import os

from gen_plasticity_mlp_sweep import PROTOCOL, baseline_lines
from gen_metaplast_branch_v2_sweep import CENTER as V2_CENTER

HERE = os.path.dirname(os.path.abspath(__file__))

# v2 diagnostics + the new turnover probe (log-only: no --plast_turnover_path, so
# no per-run .pt dumps cluttering the array; dump+plot a single chosen config
# separately with --plast_turnover_path).
PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# The most-recent carved baseline + the per-neuron budget at a middle anchor frac.
CENTER = dict(V2_CENTER)
CENTER['plast_budget'] = 'neuron'
CENTER['plast_budget_frac'] = 0.2

# OFAT axes -- headline is budget_frac x lambda on the carved baseline.
AXES = {
    # HEADLINE: per-neuron freezing budget. 1.0 = ~loose (rarely binds, ~= plain
    # v2); 0.05 = very scarce (reserves most plasticity for later tasks).
    'plast_budget_frac': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    # brake strength on the carved baseline (v2 swept 40-320); the budget should
    # let high lambda be safe by rationing how much freezing lands at once.
    'plast_lambda': [40.0, 80.0, 160.0, 320.0],
    # does the budget interact with branching? (v2: more barriers helped.)
    'plast_geo_n_barriers': [16, 32, 64],
}

# Coupled configs: the A/B control + the interesting corners.
EXTRA_CONFIGS = [
    # THE CONTROL: budget OFF on the otherwise-identical v2 center (== the plain
    # metaplast_branch_v2 anchor). The direct A/B for "does the budget help".
    {'plast_budget': 'none'},
    # Tight budget + the v2 high-lambda regime: strong-but-rationed freezing.
    {'plast_budget_frac': 0.1, 'plast_lambda': 160.0},
    {'plast_budget_frac': 0.05, 'plast_lambda': 320.0},
    # Loose budget + high lambda: should ~= unbudgeted high lambda (over-freeze).
    {'plast_budget_frac': 1.0, 'plast_lambda': 320.0},
    # Tight budget x more branching (the v2 winner direction n64) -- do limited
    # resources + many compartments stack?
    {'plast_budget_frac': 0.1, 'plast_geo_n_barriers': 64},
    # Budget on the v2 readout cross (layers=all backfired in v2; does rationing
    # the head's freezing make it survivable?).
    {'plast_budget_frac': 0.1, 'plast_layers': 'all'},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg: dict, seed: int) -> str:
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'budget_v2_sweep.txt'))
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
