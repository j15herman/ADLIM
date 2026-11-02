"""
Baseline-fill sweep, 2026-07-06.

Fills three specific baseline gaps at the standard mnistmlp protocol (SGD lr 0.03,
bs 10, 1 epoch). Hyperparameters come straight from `clser_best_args['seq-mnist']`
(the CLS-ER-tuned mnistmlp set) and are reused verbatim on seq-fmnist -- exactly
how the sgd/si/mas baselines were handled.

  * oEWC (model `ewc_on`) on seq-fmnist  -- F-MNIST regularization bar (e_lambda 90)
  * LwF                   on seq-fmnist  -- F-MNIST regularization bar
  * MAS (e_lambda 1.0)    on seq-mnist   -- the S-MNIST MAS point (strongest MAS)

= 3 configs x seeds. These produce our-own mean+/-std for the comparison tables.
Default 10 seeds to match the existing F-MNIST / S-MNIST baseline runs.

SELF-CONTAINED: reads only `clser_best_args.py` (no dependency on
gen_plasticity_mlp_sweep.baseline_lines), so it emits all three lines even on a
checkout whose baseline generator predates the oEWC/LwF addition.

Usage:
    python gen_baseline_fill_sweep.py [--seeds 0 1 2 3 4 5 6 7 8 9] [--out baseline_fill.txt]
    sbatch --array=1-$(wc -l < baseline_fill.txt)%24 \
        run_plasticity_mlp_sweep.sbatch baseline_fill.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _best_seqmnist():
    """The CLS-ER-tuned seq-mnist arg dicts (has sgd/si/mas/ewc_on/lwf/...)."""
    ns = {}
    with open(os.path.join(HERE, 'clser_best_args.py')) as f:
        exec(f.read(), ns)
    return ns['best_args']['seq-mnist']


def _flags(cfg):
    """Flatten an arg dict to a flag string; translate the older-mammoth 'wd_reg'
    key to this checkout's optimizer weight-decay flag --optim_wd (lwf)."""
    cfg = dict(cfg)
    if 'wd_reg' in cfg:
        cfg['optim_wd'] = cfg.pop('wd_reg')
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def lines_for_seed(seed, best):
    """The 3 baseline lines for one seed."""
    out = []
    # oEWC + LwF on seq-fmnist (seq-mnist-tuned args reused, like sgd/si/mas).
    out.append(f"--model ewc_on --dataset seq-fmnist {_flags(best['ewc_on'][-1])} "
               f"--enable_other_metrics 1 --seed {seed}")
    out.append(f"--model lwf --dataset seq-fmnist {_flags(best['lwf'][-1])} "
               f"--enable_other_metrics 1 --seed {seed}")
    # MAS e_lambda=1.0 on seq-mnist (the strongest MAS point).
    mas = dict(best['mas'][-1], e_lambda=1.0)
    out.append(f"--model mas --dataset seq-mnist {_flags(mas)} "
               f"--enable_other_metrics 1 --seed {seed}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'baseline_fill.txt'))
    args = ap.parse_args()

    best = _best_seqmnist()
    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    for seed in args.seeds:            # seed-outer, config-inner
        for l in lines_for_seed(seed, best):
            add(l)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out} "
          f"(3 configs x {len(args.seeds)} seeds: oEWC+LwF@fmnist, MAS1.0@mnist)")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
