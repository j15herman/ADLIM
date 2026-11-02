"""
Generate the run grid for replicating Table 1 of the CLS-ER paper
(Arani et al., ICLR 2022): Class-IL on S-MNIST / S-CIFAR-10 / S-Tiny-ImageNet
and Domain-IL on R-MNIST / P-MNIST, at buffer sizes 200/500/5120, methods
JOINT, SGD, ER, GEM, iCaRL, FDR, GSS, DER++, CLS-ER.

Hyperparameters come verbatim from the official CLS-ER fork's
utils/best_args.py (copied here as clser_best_args.py) — the same source
that produced the published numbers. Combinations the paper leaves out
('-' cells: GEM/GSS on Tiny-ImageNet, iCaRL on the Domain-IL settings)
are skipped automatically because they are absent from best_args.

Old->new Mammoth arg renames are handled below (icarl: wd_reg -> opt_wd;
softmax_temp no longer exists in the modern icarl implementation).

Usage:
    python gen_table_runs.py [--seeds 0 1 2] [--datasets ...] [--out runs.txt]
Then:
    sbatch --array=1-$(wc -l < runs.txt) run_table_array.sbatch runs.txt

For a cheap pilot (one seed, no Tiny-ImageNet) use pilot_table.sh.
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = ['seq-mnist', 'seq-cifar10', 'seq-tinyimg', 'rot-mnist', 'perm-mnist']
BUFFER_METHODS = ['er', 'gem', 'icarl', 'fdr', 'gss', 'derpp', 'clser']
BUFFERS = [200, 500, 5120]

RENAME = {'wd_reg': 'opt_wd'}   # modern icarl arg name
DROP = {'softmax_temp'}         # no longer exists in modern icarl


def load_best_args():
    ns = {}
    with open(os.path.join(HERE, 'clser_best_args.py')) as f:
        exec(f.read(), ns)
    return ns['best_args']


def flags(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        if k in DROP:
            continue
        k = RENAME.get(k, k)
        parts.append(f'--{k} {v}')
    return ' '.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                    choices=DATASETS, help='subset of datasets to generate')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'runs.txt'))
    args = ap.parse_args()

    ba = load_best_args()
    lines = []
    skipped = []

    for ds in args.datasets:
        # JOINT (upper bound) and SGD (lower bound), buffer-free.
        # best_args has no JOINT entry; use SGD's lr/batch/epochs (the
        # original protocol trains JOINT with the same budget).
        sgd_cfg = ba[ds]['sgd'][-1]
        for seed in args.seeds:
            lines.append(f'--model joint --dataset {ds} {flags(sgd_cfg)} --seed {seed}')
            lines.append(f'--model sgd --dataset {ds} {flags(sgd_cfg)} --seed {seed}')

        for method in BUFFER_METHODS:
            if method not in ba[ds]:
                skipped.append((ds, method, 'all buffers'))
                continue
            for buf in BUFFERS:
                if buf not in ba[ds][method]:
                    skipped.append((ds, method, buf))
                    continue
                cfg = ba[ds][method][buf]
                for seed in args.seeds:
                    lines.append(
                        f'--model {method} --dataset {ds} --buffer_size {buf} '
                        f'{flags(cfg)} --seed {seed}')

    with open(args.out, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'{len(lines)} runs -> {args.out}')
    if skipped:
        print('skipped (no best_args entry, matches the "-" cells in the table):')
        for s in skipped:
            print('  ', s)


if __name__ == '__main__':
    main()
