"""Plot the plasticity-turnover / freeze-timing dump from plasticity_mlp.

Consumes the blob written by `PlasticityMlp._run_turnover_probe` when a run is
launched with `--plast_turnover_probe 1 --plast_turnover_path <file>` (e.g.

    python main.py --model plasticity-mlp --plast_mlp 1 --backbone mnistmlp \
        --dataset seq-mnist --plast_lambda 20 --plast_turnover_probe 1 \
        --plast_turnover_path turnover.pt ...

then:  python scripts/plot_plasticity_turnover.py turnover.pt -o turnover.png

For each metaplastic layer it renders two panels side by side:

  (1) PLASTICITY MAP  -- the final per-weight plasticity p (the weight matrix as
      an image), so you SEE which weights are frozen (dark) vs plastic (bright).

  (2) FIRST-FREEZE MAP -- every weight coloured by WHICH TASK it first crossed
      into frozen (categorical colour, one per task), with BRIGHTNESS encoding
      WHEN WITHIN THAT TASK it froze (bright = early in the task, dark = late).
      Weights that never froze are left grey. This is the requested
      "for each task, what froze, and when within the task" view.

Pure read of the dump; no torch model needed beyond torch.load.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _task_lengths(task_start, total_steps):
    """Per-task length in observe steps, from the recorded task-start steps and
    the total steps seen so far (for the last, still-open task)."""
    starts = list(task_start)
    ends = starts[1:] + [total_steps]
    return [max(e - s, 1) for s, e in zip(starts, ends)], starts


def _freeze_rgb(freeze_step, freeze_task, task_start, total_steps, n_tasks):
    """RGB image: hue = first-freeze task (tab10), brightness = within-task
    position (early bright -> late dark). Never-frozen weights -> mid grey."""
    fs = freeze_step.numpy().astype(np.float64)
    ft = freeze_task.numpy().astype(np.int64)
    lengths, starts = _task_lengths(task_start, total_steps)

    frozen = ft >= 0
    # within-task position in [0, 1]; clamp for the still-open last task.
    pos = np.zeros_like(fs)
    for k in range(n_tasks):
        sel = ft == k
        if sel.any():
            pos[sel] = np.clip((fs[sel] - starts[k]) / lengths[k], 0.0, 1.0)

    base_grey = 0.82
    rgb = np.full(fs.shape + (3,), base_grey, dtype=np.float64)
    cmap = matplotlib.colormaps['tab10']
    for k in range(n_tasks):
        sel = ft == k
        if not sel.any():
            continue
        hue = np.array(cmap(k % 10)[:3])
        # early (pos~0) -> full-saturation colour; late (pos~1) -> darkened.
        bright = 1.0 - 0.7 * pos[sel][:, None]
        rgb[sel] = hue[None, :] * bright
    return rgb, frozen


def plot_one(dump, out, dpi):
    blob = torch.load(dump, weights_only=False, map_location='cpu')
    layers = blob['layers']
    task_start = blob['task_start_step']
    total_steps = blob['total_steps_so_far']
    n_tasks = max(len(task_start), int(blob['freeze_task'][0].max()) + 1)
    thresh = blob['freeze_thresh']

    nrows = len(layers)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.1 * nrows), squeeze=False)

    for i, name in enumerate(layers):
        p = blob['p'][i].numpy()
        rgb, frozen = _freeze_rgb(blob['freeze_step'][i], blob['freeze_task'][i],
                                  task_start, total_steps, n_tasks)

        ax_p = axes[i][0]
        im = ax_p.imshow(p, aspect='auto', cmap='viridis', vmin=0.0,
                         vmax=max(1.0, float(np.percentile(p, 99))))
        ax_p.set_title(f'{name}: plasticity map p  '
                       f'(frozen<{thresh:g}: {100 * frozen.mean():.1f}% ever)')
        ax_p.set_xlabel('presynaptic input'); ax_p.set_ylabel('postsynaptic unit')
        fig.colorbar(im, ax=ax_p, fraction=0.046, pad=0.04)

        ax_f = axes[i][1]
        ax_f.imshow(rgb, aspect='auto')
        ax_f.set_title(f'{name}: first-freeze task (colour) x within-task time '
                       f'(bright=early)')
        ax_f.set_xlabel('presynaptic input'); ax_f.set_ylabel('postsynaptic unit')

    cmap = matplotlib.colormaps['tab10']
    handles = [Patch(facecolor=cmap(k % 10), label=f'task {k}') for k in range(n_tasks)]
    handles.append(Patch(facecolor=(0.82, 0.82, 0.82), label='never frozen'))
    fig.legend(handles=handles, loc='lower center', ncol=min(n_tasks + 1, 7),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'plasticity_mlp turnover after task {blob["current_task"]}: '
                 f'freeze timing per weight', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}  (task {blob["current_task"]}, {nrows} layers, '
          f'{n_tasks} tasks, thresh={thresh:g})')


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', nargs='+',
                    help='one or more turnover .pt files written by '
                         '--plast_turnover_path (e.g. turnover.task*.pt for the '
                         'per-task snapshots). One PNG is written per input.')
    ap.add_argument('-o', '--out', default=None,
                    help='output PNG (only honoured when a single dump is given; '
                         'otherwise each dump -> <dump-without-ext>.png).')
    ap.add_argument('--dpi', type=int, default=130)
    args = ap.parse_args()

    for dump in args.dump:
        if args.out and len(args.dump) == 1:
            out = args.out
        else:
            out = os.path.splitext(dump)[0] + '.png'
        plot_one(dump, out, args.dpi)


if __name__ == '__main__':
    main()
