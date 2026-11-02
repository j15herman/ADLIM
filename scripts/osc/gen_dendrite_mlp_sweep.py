"""
Generate a sweep grid for the DENDRITIC-GEOMETRY extension of plasticity_mlp
(HANDOFF_dendrite_plasticity_mlp.md §4 step 5) on Seq-MNIST.

This is a sibling of gen_plasticity_mlp_sweep.py: same Seq-MNIST sgd protocol.
The center is RETUNED from the monotonic-lambda-20 anchor (which over-froze --
freezing is the goal but it must be reversible and capacity-preserving): now
accumulate + lambda 10 + erodable floor (accum_floor 0) + a plasticity floor
(p_min 0.001) + CONSOLIDATION-DEPENDENT STIFFNESS (Benna-Fusi: consolidated
synapses stiffen, weak ones stay labile) + neuron death (weights_line) ON.

The bet (HANDOFF §1): in-weight consolidation alone did not move Class-IL
(readout is the bottleneck). The levers here: keep the RIGHT info frozen
(stiffness locks consolidated synapses) while staying plastic enough for new
tasks (erodable floor + p_min + weight recycling). Key contrasts: death=off vs
weights_line (does recycling help?), stiffness 0 vs 0.5 vs 1 (does locking
consolidated synapses retain more?), p_min 0 vs 0.001+ (does a plasticity floor
help?), the lambda RAMP and cosine LR (freeze-late schedules), and death with
geometry=none vs conductance. Read it off the NCM oracle (does the oracle-vs-head
gap close?), the PROBE frozen/plastic fractions (is freezing now reversible?),
the GEO PROBE (eff_compartments, P_j, deaths), and task-5 trainability.

This-round axes: death knobs + p_min + stiffness + geometry contrast. Lambda
(5/10/20) and accum_floor (0/0.5/1) ablations are deferred to the next round
(noted in the AXES block).

The sgd/SI/MAS/plasticity-OFF baselines are OMITTED by default (already run in
the plasticity sweep -- compare against those); pass --with_baselines to re-emit.
Every dendrite line runs all diagnostics: NCM oracle + plasticity histogram +
the geo/death probe.

Usage:
    python gen_dendrite_mlp_sweep.py [--seeds 0] [--with_baselines]
                                     [--out dendrite_mlp_sweep.txt]
Then (reuse the plasticity array runner -- it just takes a runs file):
    sbatch --array=1-$(wc -l < dendrite_mlp_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch dendrite_mlp_sweep.txt
"""

import argparse
import os

# Reuse the plasticity sweep's protocol + baselines verbatim (same dir).
from gen_plasticity_mlp_sweep import PROTOCOL, baseline_lines

HERE = os.path.dirname(os.path.abspath(__file__))

# plasticity_mlp ON, with ALL diagnostics including the new dendrite/death probe.
DENDRITE_ON = ('--model plasticity-mlp --plast_mlp 1 '
               '--plast_ncm_oracle 1 --plast_probe 1 --plast_geo_probe 1')

# Center: the proven OSC plasticity_mlp default (accumulate + monotonic
# anti-erosion floor + lambda 20 brake, p_max 20) with GEOMETRIC spillover turned
# on at full strength. geometry needs --plast_spread > 0 (the mix fraction; 1.0 =
# the pure §2.3 kernel average). Death is OFF at the anchor and swept on its own
# axis.
CENTER = dict(
    # --- plasticity core (retuned: the monotonic lambda-20 anchor over-froze;
    #     freezing is the GOAL but it must be reversible + capacity-preserving) ---
    plast_signal='act_grad',
    plast_consistency='mean',
    plast_beta=0.99,
    plast_lambda=10.0,               # brake strength (ablate 5/20 next round)
    plast_lambda_end=-1.0,           # constant lambda (ramp tried in EXTRA_CONFIGS)
    plast_normalize=1,
    plast_p_min=0.001,               # a hair of plasticity always (no dead synapse)
    plast_p_max=20.0,                # boost-branch ceiling (accumulate)
    plast_layers='fc12',
    plast_importance_mode='accumulate',  # persistent signed accumulator
    plast_accum_floor=0.0,           # can erode below 0 (ablate 0.5/1 next round)
    plast_consol_stiffness=0.5,      # consolidated synapses stiffen (Benna-Fusi)
    plast_lr_schedule='none',        # cosine tried in EXTRA_CONFIGS
    plast_lr_min=0.0,
    plast_warmup_epochs=0.0,
    # --- geometric spillover (the new mechanism) ---
    plast_spread=1.0,                # pure geometric kernel average
    plast_geometry='conductance',    # resistor-line warped coordinate
    plast_geo_sigma=0.1,             # spillover length scale (intermediate)
    plast_geo_n_barriers=4,          # compartments per neuron
    plast_geo_barrier_height=1.0,
    plast_geo_barrier_width=0.05,
    plast_geo_bins=256,
    plast_geo_every=1,
    plast_geo_seed=0,
    # --- neuron death ON at the anchor (weight-line recycle is the focus) ---
    plast_death='weights_line',
    plast_death_thresh=0.1,          # relative: top-10% most-plastic tail
    plast_death_window=100,
    plast_death_warmup=500,
    plast_death_rate_cap=0.05,
    plast_death_relative=1,
    plast_death_reset_outgoing=1,    # Continual-Backprop outgoing reset
)

# OFAT axes (center value included for a clean anchor). DEATH-focused: the
# turnover knobs are the dense ones; geometry is trimmed to a small contrast.
AXES = {
    # death mode: weights_line (recycle weights, the focus) vs line_only (only
    # re-compartmentalize) vs off (the no-death contrast -- isolates death).
    'plast_death': ['off', 'weights_line', 'line_only'],
    # trigger: top-fraction most-plastic tail (relative). How greedy is recycling.
    'plast_death_thresh': [0.02, 0.05, 0.1, 0.2],
    # how long a neuron must stay too-plastic before it dies (transient-spike guard).
    'plast_death_window': [50, 100, 200, 400],
    # how long before death is allowed (early on everything is plastic).
    'plast_death_warmup': [200, 500, 1000],
    # max neurons recycled per step per layer (the mass-death cap).
    'plast_death_rate_cap': [0.02, 0.05, 0.1, 0.2],
    # Continual-Backprop ablation: zero the reborn unit's outgoing column or not.
    'plast_death_reset_outgoing': [0, 1],
    # geometry contrast (trimmed): none = death without compartmental spillover;
    # conductance/travel_cost = compartmented.
    'plast_geometry': ['none', 'conductance', 'travel_cost'],
    # locality of the spillover (only a few -- geometry is not the focus here).
    'plast_geo_sigma': [0.05, 0.1, 0.3],
    # compartment density.
    'plast_geo_n_barriers': [2, 4, 8],
    # plasticity floor: 0 = allow full freeze; >0 = a hair of plasticity always.
    'plast_p_min': [0.0, 0.001, 0.01, 0.1],
    # consolidation-dependent stiffness (Benna-Fusi): 0 off .. 1 max stick/labile.
    'plast_consol_stiffness': [0.0, 0.5, 1.0],
}

# NEXT ROUND (kept fixed at center this round, per the plan):
#   'plast_lambda':      [5.0, 10.0, 20.0]     # brake strength
#   'plast_accum_floor': [0.0, 0.5, 1.0]       # erosion <- -> monotonic

# Coupled A/B configs: death-centric combinations.
EXTRA_CONFIGS = [
    # Death ISOLATED from geometry: the proven plasticity default + weight
    # recycling only (no compartmental spillover). Does recycling alone help?
    {'plast_geometry': 'none', 'plast_spread': 0.0, 'plast_death': 'weights_line'},
    # Same isolation but the softer geometry-only reset has nothing to reset with
    # geometry off -> use it as a near-no-op control vs the weights_line above.
    {'plast_geometry': 'none', 'plast_spread': 0.0, 'plast_death': 'line_only'},
    # Weight recycling + travel_cost compartments (geometry A/B under death).
    {'plast_geometry': 'travel_cost', 'plast_death': 'weights_line'},
    # Aggressive recycling (kill more, sooner) -- the mass-death / capacity edge.
    {'plast_death': 'weights_line', 'plast_death_thresh': 0.2,
     'plast_death_window': 50, 'plast_death_warmup': 200,
     'plast_death_rate_cap': 0.2},
    # Gentle recycling (rare, slow) -- minimal disruption baseline.
    {'plast_death': 'weights_line', 'plast_death_thresh': 0.02,
     'plast_death_window': 400, 'plast_death_rate_cap': 0.02},
    # Absolute threshold (P_j > 1 = net-boosted/plastic) instead of the relative
    # tail -- a different death trigger entirely.
    {'plast_death': 'weights_line', 'plast_death_relative': 0,
     'plast_death_thresh': 1.0},
    # weights_line WITHOUT the outgoing reset -- clean Continual-Backprop ablation
    # (does zeroing the downstream column actually matter?).
    {'plast_death': 'weights_line', 'plast_death_reset_outgoing': 0},
    # Recycling + head protection (readout is the Class-IL bottleneck).
    {'plast_death': 'weights_line', 'plast_layers': 'all', 'plast_lambda': 30.0},
    # Recycling + dense, tight compartments (the carved regime).
    {'plast_death': 'weights_line', 'plast_geo_n_barriers': 8,
     'plast_geo_sigma': 0.05},
    # Recycling under the recency brake (ema) instead of the persistent accumulator.
    {'plast_death': 'weights_line', 'plast_importance_mode': 'ema'},
    # LAMBDA RAMP: stay plastic early, consolidate hard late (5 -> 20 over the run).
    {'plast_death': 'weights_line', 'plast_lambda': 5.0, 'plast_lambda_end': 20.0},
    # COSINE LR: anneal the learning rate to 0 over the run (simple scheduler).
    {'plast_death': 'weights_line', 'plast_lr_schedule': 'cosine',
     'plast_lr_min': 0.0},
    # Max stiffness (fully lock consolidated synapses) -- the retention extreme.
    {'plast_death': 'weights_line', 'plast_consol_stiffness': 1.0},
    # Ramp + stiffness + cosine together: the full "freeze-late, retain-hard" combo.
    {'plast_death': 'weights_line', 'plast_lambda': 5.0, 'plast_lambda_end': 20.0,
     'plast_consol_stiffness': 1.0, 'plast_lr_schedule': 'cosine'},
    # Everything on: compartments + recycling + head protection.
    {'plast_death': 'weights_line', 'plast_geometry': 'conductance',
     'plast_layers': 'all', 'plast_lambda': 30.0},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def dendrite_line(cfg: dict, seed: int) -> str:
    return f'{DENDRITE_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='seeds (default one; full seed sweep comes later)')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'dendrite_mlp_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true',
                    help='also emit the sgd/plasticity-off/si/mas baseline rows. '
                         'OFF by default -- those already ran in the plasticity '
                         'sweep; compare the dendrite runs against those results.')
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
        add(dendrite_line(dict(CENTER), seed))            # center anchor
        for axis, values in AXES.items():                 # OFAT axes
            for v in values:
                cfg = dict(CENTER)
                cfg[axis] = v
                add(dendrite_line(cfg, seed))
        for extra in EXTRA_CONFIGS:                        # coupled A/B configs
            cfg = dict(CENTER)
            cfg.update(extra)
            add(dendrite_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
