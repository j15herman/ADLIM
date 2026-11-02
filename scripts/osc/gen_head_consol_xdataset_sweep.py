"""
Cross-dataset replication of the head-consolidation A/B, 2026-06-30.
Re-runs the EXACT 8-config grid that produced the Seq-MNIST winner
(latest_sweep13) on a DIFFERENT dataset -- default Seq-Fashion-MNIST -- to check
the result transfers to harder features without touching the model.

Seq-MNIST finding being replicated (sweep13, n=11):
  layers=fc12 + head_consolidate=past_blocks + end=0.8  -> CIL 27.11  TIL 93.21  (WIN)
  layers=fc12 + head_consolidate=off        + end=0.8  -> CIL 20.07  TIL 94.68  (the wall)
  layers=all  + head_consolidate=off        + end=0.8  -> CIL 25.45  TIL 89.61  (calib trick)
  layers=all  + head_consolidate=past_blocks+ end=0.8  -> CIL 26.57  TIL 86.60  (over-mangled head)
Headline: the SURGICAL past-class-row freeze (fc12 + past_blocks) clears the
readout wall AND keeps Task-IL ~93 -- beating the layers=all importance-freeze
"calibration trick" (higher CIL AND ~5 more TIL). oracle ~65 throughout (features
untouched -> the whole gain is readout protection). Does the same picture hold on
Fashion-MNIST?

Grid (identical to sweep13):
  layers {fc12, all}
  x head_consolidate {off, past_blocks}
  x budget_frac_end {0.8, 0.85}          (frac=0.4, dthr=0.3 fixed -- the winners)
= 8 configs x seeds, on --dataset (default seq-fmnist; seq-fmnist is a drop-in
Seq-MNIST sibling -- same 5x2 class-il protocol, same mnistmlp backbone, harder
images, so the mechanism applies unchanged).

Read as a Pareto (CIL + TIL + oracle together). The transfer succeeds if
fc12+past_blocks again clears the fmnist head wall while holding Task-IL near the
fc12+off (SGD-like) TIL. Default 10 seeds (CIL variance is large; single-seed is a
coin flip). Self-contained (SWEEP5_CENTER inlined). Baselines OFF by default;
pass --with_baselines to also emit the cross-model floor/references (sgd,
plast-off, SI, MAS x3) on the SAME --dataset for the full cross-model Pareto.

Usage:
    python gen_head_consol_xdataset_sweep.py [--dataset seq-fmnist] [--seeds 0..9] [--out X.txt] [--with_baselines]
    sbatch --array=1-$(wc -l < X.txt)%24 run_plasticity_mlp_sweep.sbatch X.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: --dataset is injected by main() (default seq-fmnist), so it is NOT in
# PROTOCOL here (unlike the sibling generators which hard-code seq-mnist).
PROTOCOL = ('--backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

# The Seq-MNIST sweep4 winner, verbatim (layers overridden per-config below).
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

# The exact sweep13 axes (frac=0.4, dthr=0.3 fixed).
LAYERS = ['fc12', 'all']
HEAD = ['off', 'past_blocks']
FRAC_END = [0.8, 0.85]


def configs():
    """The 8 crossed configs as full flag dicts (deltas on SWEEP5_CENTER)."""
    out = []
    for layers in LAYERS:
        for head in HEAD:
            for frac_end in FRAC_END:
                cfg = dict(SWEEP5_CENTER)
                cfg['plast_layers'] = layers
                cfg['plast_head_consolidate'] = head
                cfg['plast_budget_frac_end'] = frac_end
                out.append(cfg)
    return out


def cfg_to_flags(cfg):
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def run_line(cfg, seed, dataset):
    return (f'{PLAST_ON} --dataset {dataset} {PROTOCOL} '
            f'{cfg_to_flags(cfg)} --seed {seed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='seq-fmnist',
                    help='Any 5x2 class-il mnistmlp dataset (seq-fmnist default; '
                         'seq-mnist to re-verify the original).')
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--with_baselines', action='store_true',
                    help='also emit the cross-model baselines (sgd / plast-off / '
                         'SI / MAS x3) on --dataset, per seed, for the full Pareto. '
                         'Hyperparameters are the seq-mnist-tuned set reused at '
                         '--dataset (no fmnist-specific tuning exists).')
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, f"head_consol_{args.dataset.replace('-', '_')}_sweep.txt")

    if args.with_baselines:                 # lazy: only this path needs the sibling
        from gen_plasticity_mlp_sweep import baseline_lines

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    cfgs = configs()
    for seed in args.seeds:                 # seed-outer, config-inner
        if args.with_baselines:
            for bl in baseline_lines(seed, args.dataset):
                add(bl)
        for cfg in cfgs:
            add(run_line(cfg, seed, args.dataset))

    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    n_bl = 6 * len(args.seeds) if args.with_baselines else 0
    print(f"wrote {len(lines)} runs to {out} "
          f"({len(cfgs)} configs x {len(args.seeds)} seeds on {args.dataset}"
          f"{f' + {n_bl} baselines' if args.with_baselines else ''})")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(out)}")


if __name__ == '__main__':
    main()
