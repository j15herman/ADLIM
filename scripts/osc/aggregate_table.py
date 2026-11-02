"""
Aggregate Mammoth logs.pyd results into a CLS-ER-paper-style table:
rows = (buffer, method), columns = datasets, cells = mean +/- std over seeds
of the FINAL mean accuracy (Class-IL for sequential datasets, Domain-IL for
rot/perm-mnist; pass --setting task-il for the Task-IL view).

Mammoth appends one python-dict line per run to
    <base_path>/<results_path>/<setting>/<dataset>/<model>/logs.pyd
(on OSC with the sbatch defaults: $DATA_ROOT/results/...). If a
(dataset, model, buffer, seed) combination was rerun, the LAST line wins.

Usage (on OSC):
    python aggregate_table.py --results_root /fs/scratch/PAS3236/mammoth_data/results
    python aggregate_table.py --results_root ... --setting task-il
    python aggregate_table.py --results_root ... --csv table.csv
"""

import argparse
import glob
import math
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

DATASET_ORDER = ['seq-mnist', 'seq-cifar10', 'seq-tinyimg', 'rot-mnist', 'perm-mnist',
                 'seq-cifar100-224']
METHOD_ORDER = ['joint', 'sgd', 'er', 'gem', 'icarl', 'fdr', 'gss', 'derpp', 'clser',
                'ewc_on', 'lwf', 'consolidation_vit']


def safe_eval(line):
    """Eval one logs.pyd line. Lines are str(dict) and may contain nan/inf
    and (numpy>=2) np.float64(...) reprs."""
    env = {'__builtins__': {}, 'nan': float('nan'), 'inf': float('inf'),
           'True': True, 'False': False, 'None': None,
           # torch.device reprs as `device(type='cuda', index=0)`
           'device': lambda type=None, index=None:
               f'{type}:{index}' if index is not None else str(type)}
    if np is not None:
        env['np'] = np
        env['array'] = np.array
    return eval(line, env)


def collect(results_root, setting):
    """-> {(dataset, model, buffer, seed): record} keeping the last line."""
    runs = {}
    pattern = os.path.join(results_root, setting, '*', '*', 'logs.pyd')
    for path in sorted(glob.glob(pattern)):
        dataset, model = path.replace('\\', '/').split('/')[-3:-1]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = safe_eval(line)
                except Exception as e:
                    print(f'! unparseable line in {path}: {e}')
                    continue
                acc_keys = [k for k in d if k.startswith('accmean_task')]
                if not acc_keys:
                    continue
                final_key = max(acc_keys, key=lambda k: int(k.split('task')[-1]))
                rec = {
                    'final_acc': float(d[final_key]),
                    'n_tasks_done': int(final_key.split('task')[-1]),
                    'forgetting': d.get('forgetting'),
                }
                buf = d.get('buffer_size')
                seed = d.get('seed')
                runs[(dataset, model, buf, seed)] = rec
    return runs


def fmt_cell(vals):
    if not vals:
        return '-'
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f'{m:.2f}'
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return f'{m:.2f}±{sd:.2f} ({len(vals)})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', type=str, default='./data/results',
                    help='the <base_path>/<results_path> folder containing class-il/, domain-il/, ...')
    ap.add_argument('--setting', type=str, default=None,
                    choices=['class-il', 'domain-il', 'task-il'],
                    help='default: class-il AND domain-il (the paper table); '
                         'task-il reads the masked-classes view of class-il runs')
    ap.add_argument('--csv', type=str, default=None, help='also write a CSV')
    args = ap.parse_args()

    settings = [args.setting] if args.setting else ['class-il', 'domain-il']
    runs = {}
    for s in settings:
        runs.update(collect(args.results_root, s))
    if not runs:
        print(f'No logs found under {args.results_root} for {settings}')
        return

    # group[(buffer, model)][dataset] -> list of final accs over seeds
    group = defaultdict(lambda: defaultdict(list))
    datasets = []
    for (dataset, model, buf, _seed), rec in sorted(runs.items()):
        group[(buf, model)][dataset].append(rec['final_acc'])
        if dataset not in datasets:
            datasets.append(dataset)
    datasets.sort(key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99)

    def row_sort(key):
        buf, model = key
        m = METHOD_ORDER.index(model) if model in METHOD_ORDER else 99
        return (buf is not None, buf if buf is not None else -1, m)

    colw = max(20, max(len(d) for d in datasets) + 2)
    header = f'{"buffer":>7} {"method":<12}' + ''.join(f'{d:>{colw}}' for d in datasets)
    print(header)
    print('-' * len(header))
    rows_csv = [['buffer', 'method'] + datasets]
    last_buf = object()
    for key in sorted(group.keys(), key=row_sort):
        buf, model = key
        if buf != last_buf:
            if last_buf is not object():
                print()
            last_buf = buf
        cells = [fmt_cell(group[key].get(d, [])) for d in datasets]
        bufs = '-' if buf is None else str(buf)
        print(f'{bufs:>7} {model:<12}' + ''.join(f'{c:>{colw}}' for c in cells))
        rows_csv.append([bufs, model] + cells)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            for r in rows_csv:
                f.write(','.join(str(x) for x in r) + '\n')
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
