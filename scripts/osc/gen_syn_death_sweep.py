"""
SINGLE-SYNAPSE death sweep -- ROUND 2 (retuned), 2026-07-02.

Tests the new `--plast_death synapse` mode (per-neuron RELATIVE + absolute floor)
against the neuron-death and death-off references, on the sweep13 WINNER center
(the current Class-IL leader): layers=fc12 + head_consolidate=past_blocks +
lambda 160 accumulate + budget=neuron ramp (frac 0.4 -> 0.8) + conductance
geometry (32 barriers). The sweep13 winner used whole-neuron death
(`--plast_death weights_line`, CIL 27.11+/-4.79 / TIL 93.21). This sweep swaps that
one knob to single-synapse turnover and tunes it.

ROUND-1 DIAGNOSIS (why this round raises thresh + lowers frac): round 1
(frac 0.05-0.2, thresh 0-5) showed PATHOLOGICAL churn -- deaths=75k-135k PER TASK
vs only 88k total fc12 synapses, i.e. every synapse recycled >=once/task. Cause: a
death->respawn->re-boost->death limit cycle. accumulate+lambda160+p_max20 BOOSTS
every below-average synapse to p~20; synapse death targets the most-plastic = that
boosted pool (huge); a reborn synapse (reset p=1, importance=0) gets the ~1e6 x
Benna-Fusi stiffness gain and re-boosts to the p_max ceiling within a step, then
re-qualifies after `window`. thresh 1->5 barely changed deaths because the boost
ceiling is 20, so any thresh << 20 gates nothing. Fix here: push thresh toward 20
(only ceiling-PINNED, truly never-consolidating synapses die) and shrink frac.

Single-synapse death recycles INDIVIDUAL synapses instead of whole neurons: within
EACH neuron, the synapses that are (a) in that neuron's top --plast_death_syn_frac
most plastic AND (b) above the absolute floor --plast_death_syn_thresh (on the EMA
per-synapse plasticity, range [p_min, p_max]=[0.001, 20]), that stay so for
--plast_death_window steps, get their weight reinit + position/u respawned on the
same dendrite + importance zeroed + p=1 (no outgoing reset). The floor is the churn
brake: it must sit near p_max to exclude the ordinary boosted pool.

Grid (OFAT deltas on SYN_CENTER, deduped, + 3 death-mode anchors):

  death-mode anchors:  off | weights_line (=the sweep13 winner) | synapse (center)
  syn_frac      {0.02, 0.05*, 0.1}        top-% per neuron (LOWER than round 1)
  syn_thresh    {10, 12, 15*, 18}         absolute EMA-p floor toward p_max=20
  window        {100*, 200}               consecutive steps
  rate_cap      {0.02, 0.05*}             max synapses/step PER NEURON (frac of fan-in)
  crosses       (frac0.02 x thr18 = most conservative), (frac0.1 x thr10 = bracket)

(* = center value.) 12 configs x seeds. All diagnostics ON. Report CIL + TIL +
oracle together (Pareto). PRIMARY read this round: does `plast SYN DEATH task T:
synapse_deaths=...` drop to O(1k) or less (vs round-1's ~90k) AND does CIL move off
the ~20 wall toward the neuron-death bar (26.7)? If deaths are still huge at
thresh18, the loop needs the code-side fix (respawn refractory / don't reset reborn
p=1), not just knobs. Default 5 seeds; bump to 10 for anything decisive
(single-seed CIL std is 3-6).

REQUIRES the model sync: `--plast_death synapse`, `--plast_death_syn_frac`,
`--plast_death_syn_thresh` are NEW flags (models/plasticity_mlp.py +
models/plasticity_mlp_utils/dendrite_geometry.py must be on OSC first).

Self-contained (SYN_CENTER inlined verbatim). Baselines OFF by default.

Usage:
    python gen_syn_death_sweep.py [--seeds 0 1 2 3 4] [--out syn_death_sweep.txt]
    sbatch --array=1-$(wc -l < syn_death_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch syn_death_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# The sweep13 WINNER (layers=fc12 + past_blocks + budget_frac_end=0.8), copied
# verbatim from its run args, with death switched to the new single-synapse mode.
# The neuron-death flags (thresh/relative/reset_outgoing) are kept but UNUSED in
# synapse mode; the two 'off'/'weights_line' anchors override plast_death to reuse
# them. syn_frac/syn_thresh/window/warmup/rate_cap drive the synapse mode.
SYN_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='fc12',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='synapse', plast_death_thresh=0.3, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_death_syn_frac=0.05, plast_death_syn_thresh=15.0,
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.8,
    plast_head_consolidate='past_blocks',
)

# OFAT deltas on SYN_CENTER (+ death-mode anchors). Center value omitted (it IS the
# center); dedup drops any accidental repeats.
DELTAS = [
    # death-mode anchors (the turnover-strategy A/B/C)
    ('death_off',          {'plast_death': 'off'}),
    ('death_neuron',       {'plast_death': 'weights_line'}),   # == the sweep13 winner
    ('syn_center',         {}),                                # frac0.05 thresh15
    # top-% per neuron -- LOWER than round 1 (0.05/0.1/0.2 churned); 0.05 is center
    ('syn_frac0.02',       {'plast_death_syn_frac': 0.02}),
    ('syn_frac0.1',        {'plast_death_syn_frac': 0.1}),
    # absolute EMA-p floor -- RAISED toward p_max=20 (round-1 thresh 1/5 didn't gate
    # the boosted pool at all: thresh must approach the ceiling to shrink eligibility)
    ('syn_thresh10',       {'plast_death_syn_thresh': 10.0}),
    ('syn_thresh12',       {'plast_death_syn_thresh': 12.0}),
    ('syn_thresh18',       {'plast_death_syn_thresh': 18.0}),
    # window / rate cap (extra throttles)
    ('syn_window200',      {'plast_death_window': 200}),
    ('syn_ratecap0.02',    {'plast_death_rate_cap': 0.02}),
    # crosses: most conservative (kills only ceiling-pinned, few per neuron) and a
    # slightly-more-aggressive bracket
    ('syn_frac0.02_thr18', {'plast_death_syn_frac': 0.02, 'plast_death_syn_thresh': 18.0}),
    ('syn_frac0.1_thr10',  {'plast_death_syn_frac': 0.1,  'plast_death_syn_thresh': 10.0}),
]


def configs():
    """Full flag dicts (each delta applied on SYN_CENTER)."""
    return [dict(SYN_CENTER, **delta) for _label, delta in DELTAS]


def cfg_to_flags(cfg):
    # Emit NEGATIVE values as `--flag=value`. A bare "-1.0" token trips mammoth's
    # check_multiple_defined_arg_during_string_parse (it treats any '-'-prefixed
    # token as a flag name, so two negative-valued flags -- e.g. lambda_end=-1.0
    # and budget_frac_end=-1.0 -- look like the flag "-1.0" defined twice). The
    # '=' form keeps the flag name as the token, so it can never collide.
    out = []
    for k, v in cfg.items():
        s = str(v)
        out.append(f'--{k}={s}' if s.startswith('-') else f'--{k} {s}')
    return ' '.join(out)


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'syn_death_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true')
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    if args.with_baselines:                 # lazy: only this path needs the sibling
        from gen_plasticity_mlp_sweep import baseline_lines

    cfgs = configs()
    for seed in args.seeds:                 # seed-outer, config-inner
        if args.with_baselines:
            for bl in baseline_lines(seed):
                add(bl)
        for cfg in cfgs:
            add(run_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"({len(cfgs)} configs x {len(args.seeds)} seeds"
          f"{' + baselines' if args.with_baselines else ''})")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()