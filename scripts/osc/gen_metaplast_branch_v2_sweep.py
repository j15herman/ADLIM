"""
Round-2 targeted sweep for plasticity_mlp on Seq-MNIST: ablate the new
best-so-far baseline distilled from the round-1 metaplast/branching sweep
(slurm job 6054797, 2026-06-24). Sibling of gen_metaplast_branch_sweep.py.

What round 1 established (see that file + the dendrite memory):
  * Class-IL is still pinned ~19-20% (readout wall); the only configs that
    nudged it to ~22% (matching MAS e_lambda 0.1 = 22.63 and BEATING it on
    Task-IL) were the AGGRESSIVE-LOCAL-COMPARTMENT ones -- run 41 (conductance
    force-carve: h=8, w=0.2, sigma=0.02, n=16 -> eff_comp 432/92) and run 22
    (conductance n=64). Branching FINALLY engaged (eff_comp 5->430, no longer 1).
  * consol_stiffness=2.0 was the sweet spot (run 8, CIL 21.6); 4.0 over-froze
    (TIL crashed to 83, frozen frac 0.75).
  * Higher lambda raised the NCM oracle + Task-IL and stayed stable even at 320
    (frozen weights are robust); lambda 80 is the chosen new floor.
  * stiff_max 1000 helped; more barriers + smaller sigma helped monotonically.
  * layers=all BACKFIRED everywhere (braking the head crashes Task-IL to 75-88
    and drops the oracle); kept OUT of the baseline, one check retained.

NEW BASELINE (this file's CENTER) = run 41's force-carved geometry + the winners:
  lambda 80, consol_stiffness 2.0, stiff_max 1000, geometry=conductance with
  barrier_height 8 / barrier_width 0.2 / geo_sigma 0.02 and n_barriers BUMPED to
  32 (more branching, exp-39 density) -> expect eff_comp in the hundreds. fc12,
  cosine LR, weight-recycling death, accumulate.

Then OFAT-ablate every knob baked into that baseline: lambda, consol_stiffness,
stiff_max, stiff_min, geo_n_barriers, geo_sigma, geo_barrier_height,
geo_barrier_width, geometry. Read off the same diagnostics (NCM oracle vs head,
PROBE frozen/plastic, GEO PROBE eff_compartments/deaths). Bar: MAS (CIL 22.6 /
TIL 94.7) and the round-1 best (run 41: CIL 22.6 / TIL 96.6 / oracle 69.9).

Baselines OMITTED (compare to prior runs); --with_baselines to re-emit.

Usage:
    python gen_metaplast_branch_v2_sweep.py [--seeds 0] [--out metaplast_branch_v2_sweep.txt]
    sbatch --array=1-$(wc -l < metaplast_branch_v2_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch metaplast_branch_v2_sweep.txt
"""

import argparse
import os

from gen_plasticity_mlp_sweep import PROTOCOL, baseline_lines

HERE = os.path.dirname(os.path.abspath(__file__))

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 '
            '--plast_ncm_oracle 1 --plast_probe 1 --plast_geo_probe 1')

# NEW BASELINE = run 41 force-carve + round-1 winners (lambda 80, stiffness 2,
# stiff_max 1000) with branching bumped to n=32.
CENTER = dict(
    # --- plasticity core ---
    plast_signal='act_grad',
    plast_consistency='mean',
    plast_beta=0.99,
    plast_lambda=80.0,               # round-1 chosen floor (was 20)
    plast_lambda_end=-1.0,
    plast_normalize=1,
    plast_p_min=0.001,
    plast_p_max=20.0,
    plast_layers='fc12',             # all backfired in round 1; one cross retained
    plast_importance_mode='accumulate',
    plast_accum_floor=0.0,
    plast_accum_center=1,
    # --- meta-metaplasticity (stiffness 2 sweet spot, stiff_max 1000 helped) ---
    plast_consol_stiffness=2.0,
    plast_stiff_min=0.01,
    plast_stiff_max=1000.0,
    # --- LR ---
    plast_lr_schedule='cosine',
    plast_lr_min=0.0,
    plast_warmup_epochs=0.0,
    # --- dendritic geometry: run-41 force-carve + MORE branching (n 16 -> 32) ---
    plast_spread=1.0,
    plast_geometry='conductance',
    plast_geo_sigma=0.02,            # very local
    plast_geo_n_barriers=32,         # bumped from run-41's 16
    plast_geo_barrier_height=8.0,    # tall: forces the warped coord to carve
    plast_geo_barrier_width=0.2,     # wide bumps (run 41)
    plast_geo_bins=256,
    plast_geo_every=1,
    plast_geo_seed=0,
    # --- neuron death ON (weight-line recycle) ---
    plast_death='weights_line',
    plast_death_thresh=0.1,
    plast_death_window=100,
    plast_death_warmup=500,
    plast_death_rate_cap=0.05,
    plast_death_relative=1,
    plast_death_reset_outgoing=1,
)

# OFAT axes -- ablate every knob baked into the new baseline (center included).
AXES = {
    # brake strength: re-find the optimum under the new (carved) geometry.
    'plast_lambda': [40.0, 80.0, 160.0, 320.0],
    # consolidation stiffness: 2 sweet spot in r1; does 3 over-freeze here too?
    'plast_consol_stiffness': [1.0, 2.0, 3.0, 4.0],
    # lability of weak synapses' importance (1000 helped in r1).
    'plast_stiff_max': [100.0, 300.0, 1000.0, 3000.0],
    # how hard consolidated synapses lock importance.
    'plast_stiff_min': [0.001, 0.01, 0.1],
    # BRANCHING: push compartment count further (more helped monotonically).
    'plast_geo_n_barriers': [16, 32, 64, 128],
    # LOCALITY: push spillover length scale down (smaller helped).
    'plast_geo_sigma': [0.005, 0.01, 0.02, 0.05],
    # barrier height: how hard the coordinate carves (8 = run 41).
    'plast_geo_barrier_height': [2.0, 4.0, 8.0, 16.0],
    # barrier width: bump extent (0.2 = run 41).
    'plast_geo_barrier_width': [0.05, 0.1, 0.2, 0.4],
    # geometry kind: conductance (carve) vs travel_cost (discrete jumps).
    'plast_geometry': ['conductance', 'travel_cost'],
}

# Coupled configs: the round-1 winners combined + a few crosses.
EXTRA_CONFIGS = [
    # travel_cost analogue of the carve baseline (exp-39 density) under the new
    # lambda/stiffness -- does discrete-jump geometry match conductance now?
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 32,
     'plast_geo_barrier_height': 1.0, 'plast_geo_sigma': 0.02},
    # ULTRA branch + locality: push everything past run 41.
    {'plast_geo_n_barriers': 128, 'plast_geo_sigma': 0.005,
     'plast_geo_barrier_height': 16.0},
    # Max carve, travel_cost flavor (run-40-like, scaled up).
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 64,
     'plast_geo_sigma': 0.01, 'plast_geo_barrier_height': 4.0},
    # Best-of cross: higher lambda x more barriers.
    {'plast_lambda': 160.0, 'plast_geo_n_barriers': 64},
    # Stiffness 1 (less freeze) at high branch -- is 2 still better when carved?
    {'plast_consol_stiffness': 1.0, 'plast_geo_n_barriers': 64},
    # Late-freeze ramp on the carved baseline.
    {'plast_lambda': 10.0, 'plast_lambda_end': 80.0},
    # ONE readout check: does the high-branch + reversible-stiffness regime make
    # layers=all survivable this time? (r1: all backfired hard.)
    {'plast_layers': 'all'},
    # readout check + late freeze so early tasks still train the head first.
    {'plast_layers': 'all', 'plast_lambda': 10.0, 'plast_lambda_end': 80.0},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg: dict, seed: int) -> str:
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'metaplast_branch_v2_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true',
                    help='also emit sgd/SI/MAS baseline rows (OFF by default).')
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line)
            lines.append(line)

    for seed in args.seeds:
        if args.with_baselines:
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
    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
