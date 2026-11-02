"""
Targeted follow-up sweep for plasticity_mlp on Seq-MNIST: attack the Class-IL
wall directly. Sibling of gen_dendrite_mlp_sweep.py (same protocol/baselines).

Motivation (from the 3-log read vs MAS, 2026-06-24):
  * Every run -- sgd, SI, MAS, and all plasticity_mlp variants -- hits FLAT ZERO
    Class-IL on every task except the most recent (raw CIL like [0,0,0,~0,~95]).
    The head collapses onto the last task; the NCM oracle sits at 59-66% CIL
    while the actual head delivers ~20%. The readout, not the consolidation rule,
    is the wall. This has now reproduced across reruns.
  * The best variant (run 37: stiffness 1.0, lambda 5->20 ramp, cosine LR, death)
    beat MAS on Task-IL (97.2 vs 94.7) but did nothing for Class-IL.
  * The dendritic geometry was INERT in all runs: every GEO PROBE reported
    eff_comp ~ 1.00. With barrier_height 1.0 / width 0.05 / sigma 0.1 the
    conductance warped coordinate barely deviates from x, so spillover is
    effectively uniform -- the compartmentalization never engaged.

Four lever clusters this round (newfound stability with frozen weights lets us
push much harder than the lambda-10/20 regime):

  (1) LAMBDA JUICING. Crank the brake (20 -> 320). Stable frozen weights mean we
      can over-consolidate without blowup. ON fc12 this only buys Task-IL; the
      juiced-lambda x layers=all crosses (EXTRA) are what can actually move
      Class-IL, because the head is the only path to it.
  (2) META-METAPLASTICITY (Benna-Fusi consolidation-dependent stiffness): how
      hard consolidated synapses LOCK their importance (consol_stiffness,
      stiff_min) vs how labile weak synapses stay (stiff_max). Never swept before
      -- stiff_min/stiff_max were pinned at center.
  (3) LR SCHEDULING. Cosine helped Task-IL; sweep the floor (lr_min), the
      schedule on/off, warmup-then-cosine, and cosine x juiced lambda.
  (4) BRANCHING / LOCALITY. Make the dendrite ACTUALLY branch and keep spillover
      VERY LOCAL. eff_comp = 1 + (#consecutive-synapse u-gaps > geo_sigma), so
      real compartments need barrier jumps that exceed geo_sigma. travel_cost
      (discrete full-height jumps) guarantees a break per barrier; small geo_sigma
      keeps spillover inside the compartment. Moderate (n_barriers 8-16) to great
      (32-64) branching, sigma down to 0.01.

Readouts per line (all probes on): plast NCM ORACLE (head-independent CIL/TIL
ceiling -- does the oracle-vs-head gap close?), plast PROBE (frozen/plastic
fractions -- is the freeze reversible?), plast GEO PROBE (eff_compartments -- did
branching finally engage? -- P_j, deaths).

Baselines are OMITTED (the MAS/SI/sgd rows already ran; the MAS e_lambda 1.0/10.0
logs were truncated and need a config fix -- handled separately). Pass
--with_baselines to re-emit.

Usage:
    python gen_metaplast_branch_sweep.py [--seeds 0] [--out metaplast_branch_sweep.txt]
Then (reuse the plasticity array runner -- it just takes a runs file):
    sbatch --array=1-$(wc -l < metaplast_branch_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch metaplast_branch_sweep.txt
"""

import argparse
import os

# Reuse the plasticity sweep's protocol + baselines verbatim (same dir).
from gen_plasticity_mlp_sweep import PROTOCOL, baseline_lines

HERE = os.path.dirname(os.path.abspath(__file__))

# plasticity_mlp ON with ALL diagnostics (NCM oracle + histogram + geo/death probe).
PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 '
            '--plast_ncm_oracle 1 --plast_probe 1 --plast_geo_probe 1')

# Center = the best-so-far config (run 37), made into a clean constant-lambda
# anchor: accumulate + erodable floor, lambda 20 (the new floor -- already juiced
# from 10), max-ish consolidation stiffness, cosine LR, weight-recycling death ON.
# Geometry is the prior (inert) conductance setting so the BRANCHING axes show
# the effect of actually carving compartments relative to the eff_comp~1 anchor.
CENTER = dict(
    # --- plasticity core ---
    plast_signal='act_grad',
    plast_consistency='mean',
    plast_beta=0.99,
    plast_lambda=20.0,               # constant brake (juiced axis pushes this up)
    plast_lambda_end=-1.0,           # constant; ramp lives in EXTRA_CONFIGS
    plast_normalize=1,
    plast_p_min=0.001,
    plast_p_max=20.0,
    plast_layers='fc12',             # readout cross lives in the axis + EXTRA
    plast_importance_mode='accumulate',
    plast_accum_floor=0.0,
    plast_accum_center=1,
    # --- meta-metaplasticity (Benna-Fusi consolidation-dependent stiffness) ---
    plast_consol_stiffness=1.0,      # consolidated synapses stiffen hard
    plast_stiff_min=0.01,            # gain when fully consolidated (locks importance)
    plast_stiff_max=100.0,           # gain when fully plastic (labile importance)
    # --- LR schedule ---
    plast_lr_schedule='cosine',
    plast_lr_min=0.0,
    plast_warmup_epochs=0.0,
    # --- dendritic geometry (prior, inert setting -- branching axes move off it) ---
    plast_spread=1.0,
    plast_geometry='conductance',
    plast_geo_sigma=0.1,
    plast_geo_n_barriers=4,
    plast_geo_barrier_height=1.0,
    plast_geo_barrier_width=0.05,
    plast_geo_bins=256,
    plast_geo_every=1,
    plast_geo_seed=0,
    # --- neuron death ON (weight-line recycle, run-37 setting) ---
    plast_death='weights_line',
    plast_death_thresh=0.1,
    plast_death_window=100,
    plast_death_warmup=500,
    plast_death_rate_cap=0.05,
    plast_death_relative=1,
    plast_death_reset_outgoing=1,
)

# OFAT axes (center value included for a clean anchor).
AXES = {
    # (1) LAMBDA JUICING -- stable frozen weights let us over-consolidate.
    'plast_lambda': [20.0, 40.0, 80.0, 160.0, 320.0],
    # (2) META-METAPLASTICITY: stick of consolidated / lability of weak synapses.
    'plast_consol_stiffness': [0.0, 0.5, 1.0, 2.0, 4.0],
    'plast_stiff_min': [0.0001, 0.001, 0.01, 0.1],   # smaller = locks harder
    'plast_stiff_max': [10.0, 100.0, 1000.0],        # larger = weak synapses freer
    # (3) LR SCHEDULING.
    'plast_lr_schedule': ['none', 'cosine'],
    'plast_lr_min': [0.0, 0.003, 0.01],
    # (4) BRANCHING / LOCALITY -- carve REAL compartments (prior eff_comp ~ 1).
    'plast_geometry': ['conductance', 'travel_cost'],
    'plast_geo_n_barriers': [4, 8, 16, 32, 64],       # moderate -> great branching
    'plast_geo_barrier_height': [1.0, 2.0, 4.0, 8.0], # tall enough to exceed sigma
    'plast_geo_sigma': [0.01, 0.02, 0.05, 0.1],       # SMALL = very local spillover
    'plast_geo_barrier_width': [0.05, 0.1, 0.2],
    # (5) THE READOUT LEVER -- the only path to Class-IL.
    'plast_layers': ['fc12', 'all'],
}

# Coupled A/B configs: the crosses that actually target Class-IL + real branching.
EXTRA_CONFIGS = [
    # --- juiced lambda must reach the HEAD to move Class-IL (fc12 brake can't) ---
    {'plast_layers': 'all', 'plast_lambda': 40.0},
    {'plast_layers': 'all', 'plast_lambda': 80.0},
    {'plast_layers': 'all', 'plast_lambda': 160.0},
    {'plast_layers': 'all', 'plast_lambda': 320.0},
    # Freeze the head LATE (ramp) so early tasks still train the readout, then lock.
    {'plast_layers': 'all', 'plast_lambda': 10.0, 'plast_lambda_end': 80.0,
     'plast_lr_schedule': 'cosine'},
    {'plast_layers': 'all', 'plast_lambda': 10.0, 'plast_lambda_end': 160.0},
    # --- REAL branching: travel_cost discrete barriers, tall, VERY local ---
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 16,
     'plast_geo_barrier_height': 1.0, 'plast_geo_sigma': 0.02},          # moderate
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 32,
     'plast_geo_barrier_height': 1.0, 'plast_geo_sigma': 0.02},          # great
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 64,
     'plast_geo_barrier_height': 2.0, 'plast_geo_sigma': 0.01},          # great + ultra-local
    # Force CONDUCTANCE to actually carve: tall + wide bumps, small sigma.
    {'plast_geometry': 'conductance', 'plast_geo_barrier_height': 8.0,
     'plast_geo_barrier_width': 0.2, 'plast_geo_sigma': 0.02,
     'plast_geo_n_barriers': 16},
    # Branching + head protection together (local compartments + readout brake).
    {'plast_geometry': 'travel_cost', 'plast_geo_n_barriers': 32,
     'plast_geo_sigma': 0.02, 'plast_layers': 'all', 'plast_lambda': 80.0},
    # --- meta-metaplasticity extreme: widest stick/labile gap ---
    {'plast_consol_stiffness': 2.0, 'plast_stiff_min': 0.0001,
     'plast_stiff_max': 1000.0},
    # --- LR: warmup-then-cosine, and cosine x juiced lambda ---
    {'plast_lr_schedule': 'cosine', 'plast_warmup_epochs': 0.5},
    {'plast_lr_schedule': 'cosine', 'plast_lambda': 80.0},
    # --- the full "retain-hard, branch-local, protect-head" combo ---
    {'plast_layers': 'all', 'plast_lambda': 80.0, 'plast_consol_stiffness': 2.0,
     'plast_lr_schedule': 'cosine', 'plast_geometry': 'travel_cost',
     'plast_geo_n_barriers': 32, 'plast_geo_sigma': 0.02},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg: dict, seed: int) -> str:
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='seeds (default one; full seed sweep comes later)')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'metaplast_branch_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true',
                    help='also emit sgd/SI/MAS baseline rows. OFF by default -- '
                         'those already ran (compare against those results).')
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
        for extra in EXTRA_CONFIGS:                         # coupled A/B configs
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
