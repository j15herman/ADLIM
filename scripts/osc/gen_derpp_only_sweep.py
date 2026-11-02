"""
DER++ ALONE (no metaplastic brake) on Seq-MNIST -- the replay baseline, properly swept.

WHY: in the brake x replay study (gen_derpp_plast_sweep.py) the `derpp` arm is carried
at a SINGLE hyperparameter point (buffer 500, alpha 1.0, beta 0.5 -- the mammoth
Seq-MNIST values from clser_best_args.py). That is enough for an apples-to-apples
control, but it does NOT tell you how good replay ALONE can get, or how sensitive it
is to its own dials. That matters for the headline claim:

  - if a TUNED DER++ matches derpp_plast, the brake is redundant on top of replay;
  - if derpp_plast still wins at DER++'s own best point, the brake buys something real.

Dials swept (all on the standard protocol: sgd, lr 0.03, batch 10, 1 epoch/task):

  --buffer_sizes  reservoir capacity (the dominant dial)
  --alphas        logit-distillation weight -- the "dark experience" MSE term
  --betas         label-replay CE weight (beta 0 = pure DER, logits only)
  --minibatch     replay batch drawn per term (default 10 = the task batch size)

Many seeds by default: single-seed Class-IL on this project is a task-3 coin flip
(std 3-6), so every point runs 10 seeds and you read mean +/- std TOGETHER.

`--model derpp` is STOCK mammoth (models/derpp.py, shared --buffer_size/--alpha/--beta
from add_rehearsal_args). Nothing here touches models/plasticity_mlp.py, so this sweep
is unaffected by the plasticity model file or its registry entry.

Self-contained (no sibling imports) -- copy just this file to the OSC checkout.

Usage:
    python gen_derpp_only_sweep.py                       # 4 alpha x 2 beta x 10 seeds = 80
    python gen_derpp_only_sweep.py --buffer_sizes 200 500 5120
    python gen_derpp_only_sweep.py --with_control        # + the sgd floor, one per seed
    python gen_derpp_only_sweep.py --seeds 0 1 2         # quick look
    sbatch --array=1-$(wc -l < derpp_only_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_only_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Identical training protocol to every other arm of the study, so a DER++ point here
# is directly comparable to the derpp/derpp_plast arms of gen_derpp_plast_sweep.py.
# --enable_other_metrics 1 turns on Forgetting/BWT/FWT in the .err log.
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')


def derpp_line(buffer_size, minibatch, alpha, beta, seed):
    """One stock DER++ run. --buffer_size/--alpha/--beta are required by models/derpp.py."""
    return (f'--model derpp {PROTOCOL} '
            f'--buffer_size {buffer_size} --minibatch_size {minibatch} '
            f'--alpha {alpha} --beta {beta} --seed {seed}')


def sgd_line(seed):
    """Control: no replay, no brake (the forgetting floor)."""
    return f'--model sgd {PROTOCOL} --seed {seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--buffer_sizes', type=int, nargs='+', default=[500],
                    help='Reservoir capacities to sweep. Default [500] (the value the '
                         'brake x replay study uses). Try 200 500 5120 for the classic '
                         'mammoth buffer ladder.')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.5, 1.0],
                    help='Logit-distillation (MSE) weights.')
    ap.add_argument('--betas', type=float, nargs='+', default=[0.5, 1.0],
                    help='Label-replay (CE) weights. Include 0.0 for pure DER.')
    ap.add_argument('--minibatch', type=int, default=10,
                    help='Replay minibatch per term (default 10 = --batch_size).')
    ap.add_argument('--with_control', action='store_true',
                    help='Also emit the sgd floor once per seed, for reference.')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_only_sweep.txt'))
    args = ap.parse_args()

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line)
            lines.append(line)

    n_points = 0
    for seed in args.seeds:                       # seed-outer: cancelling early still
        if args.with_control:                     # leaves every point at fewer seeds
            add(sgd_line(seed))
        for bs in args.buffer_sizes:
            for alpha in args.alphas:
                for beta in args.betas:
                    add(derpp_line(bs, args.minibatch, alpha, beta, seed))
    n_points = len(args.buffer_sizes) * len(args.alphas) * len(args.betas)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"  {n_points} DER++ points ({len(args.buffer_sizes)} buffer x "
          f"{len(args.alphas)} alpha x {len(args.betas)} beta) x {len(args.seeds)} seeds"
          + (" + sgd control" if args.with_control else ""))
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
