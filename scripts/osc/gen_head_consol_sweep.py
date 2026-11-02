"""
Per-class-block HEAD consolidation sweep, 2026-06-30
(HANDOFF_head_consolidation). REBUILT on the sweep4/budget-ramp WINNER center
(the current Class-IL leader), not the old fc12/lambda-80 v2 center.

Why the pivot: the latest 10-seed round (layers=all, lambda 160, accumulate,
budget=neuron with a budget_frac ramp + death) is the first family to clear the
~19-21 readout wall — best configs sit at Class-IL ~25. Its winners:

  frac=0.4 budget_frac_end=0.85 dthr=0.3  -> CIL 25.06+/-6.15  TIL 89.45  oracle 64.6
  frac=0.4 budget_frac_end=0.80 dthr=0.3  -> CIL 25.05+/-3.51  TIL 89.62  oracle 65.5  (tightest)
  frac=0.75 (no ramp)          dthr=0.3   -> CIL 23.61+/-5.90  TIL 88.25  oracle 64.8
(dthr=0.3 beats dthr=0.2 across the board; the frac-0.4 ramps lead.)

BUT that CIL gain is a HEAD-CALIBRATION TRICK: layers=all lets importance freeze
the WHOLE classifier, redistributing logit mass off the last task (per-task @t5
CIL goes [0,0,0,~5,~96]->[~2,~8,~8,~16,~93]) -- it de-biases the readout rather
than protecting features. The tell: Task-IL CRASHES 95->~89 (below SGD's 94.94
floor) and the NCM oracle DROPS to ~64 (vs ~68 at fc12). So we are trading Task-IL
for Class-IL by mangling the head.

Head-consolidation is the SURGICAL alternative: instead of importance-freezing the
entire head, `--plast_head_consolidate past_blocks` hard-freezes ONLY the
weight+bias rows of PAST classes each step (class < get_offsets(current_task)[0]),
leaving the current task's rows plastic. That directly stops the recency erosion
of old-class logits WITHOUT touching within-task (Task-IL) decisions. The bet:

  (1) On the winning layers=all center, past_blocks pushes CIL past ~25 and/or
      RECOVERS Task-IL (the past rows stop being importance-mangled).
  (2) With layers=fc12 (head NOT importance-frozen -> TIL should stay ~95),
      past_blocks alone recovers much of the layers=all CIL gain -- i.e. gets the
      Class-IL lift WITHOUT the Task-IL crash. This is the headline isolation.

Grid (paired so the head off/past_blocks A/B is exact):

  head_consolidate {off, past_blocks}        the A/B
  x center {top1 end0.85, top2 end0.80, best flat0.75}   the CIL leaders (all dthr=0.3)
  x plast_layers {all, fc12}                 head importance-frozen (all, the trick)
                                             vs head plastic except the hard row-freeze
                                             (fc12, the clean test)

= 2x3x2 = 12 configs x seeds. All diagnostics ON. Report CIL + TIL + oracle
TOGETHER (Pareto) -- the win is CIL up AND TIL recovered, not CIL alone. Decisive
reads: does the past_blocks x fc12 cell reach the layers=all CIL while holding
TIL ~95, and does the Raw Class-IL vector lift earlier task blocks off zero.
Default 10 seeds (the round these winners came from is n=10; single-seed CIL is a
tail coin-flip -- std 3-6).

Self-contained (SWEEP5_CENTER inlined verbatim -- no sibling imports, runs on the
OSC checkout). Baselines OFF by default (--with_baselines).

Usage:
    python gen_head_consol_sweep.py [--seeds 0 1 2 3 4 5 6 7 8 9] [--out head_consol_sweep.txt]
    sbatch --array=1-$(wc -l < head_consol_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch head_consol_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

# The sweep4/budget-ramp WINNER, copied verbatim from the top-1 run's args
# (frac=0.4 budget_frac_end=0.85 dthr=0.3): layers=all, lambda 160, accumulate,
# conductance geometry (32 barriers), weights_line death, budget=neuron with the
# ramp. This is the current Class-IL leader (~25). The three WINNER_CENTERS below
# vary only the budget frac / ramp end / death threshold off this base.
SWEEP5_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='all',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='weights_line', plast_death_thresh=0.3, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.85,
)

PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# The CIL-leading budget/death configs from the 10-seed round (all dthr=0.3, which
# beat dthr=0.2). Each is a delta on SWEEP5_CENTER; the flat one turns the ramp off.
WINNER_CENTERS = [
    # (label, delta)
    ('top1_end0.85',  {'plast_budget_frac': 0.4,  'plast_budget_frac_end': 0.85,
                       'plast_death_thresh': 0.3}),                    # CIL 25.06
    ('top2_end0.80',  {'plast_budget_frac': 0.4,  'plast_budget_frac_end': 0.80,
                       'plast_death_thresh': 0.3}),                    # CIL 25.05, tightest
    ('flat0.75',      {'plast_budget_frac': 0.75, 'plast_budget_frac_end': -1.0,
                       'plast_death_thresh': 0.3}),                    # CIL 23.61, no ramp
]

HEAD = ['off', 'past_blocks']     # the A/B
LAYERS = ['all', 'fc12']          # head importance-frozen (trick) vs head plastic + hard row-freeze


def configs():
    """The 12 crossed configs as full flag dicts (deltas applied on SWEEP5_CENTER)."""
    out = []
    for _label, delta in WINNER_CENTERS:
        for head in HEAD:
            for layers in LAYERS:
                cfg = dict(SWEEP5_CENTER)
                cfg.update(delta)
                cfg['plast_head_consolidate'] = head
                cfg['plast_layers'] = layers
                out.append(cfg)
    return out


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed):
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'head_consol_sweep.txt'))
    ap.add_argument('--with_baselines', action='store_true')
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    baseline_lines = None
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
