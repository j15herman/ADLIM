"""
DER++ x metaplastic-brake sweep, 2026-07-08. Arm 1 of the brake x replay x
regularizer VARIANCE study (see memory plasticity-mlp / the DER++ combine that
just landed in models/plasticity_mlp.py).

QUESTION: the brain fights catastrophic forgetting with BOTH consolidation and
episodic replay. Does adding replay (DER++) to the per-weight metaplastic brake
lift Class-IL and -- the real target -- CUT the seed variance? Single-seed CIL on
this project is a task-3 coin flip (std 3-6), so every arm runs many seeds and we
read mean +/- std TOGETHER.

The three requested arms (replay, replay+brake, replay+brake+head-freeze) plus the
matched no-replay controls, because the replay DELTA is only measurable against the
same config without replay:

  sgd                 neither  (floor / control)                 [--with_controls]
  derpp               REPLAY alone                               *** requested ***
  plast               brake alone, head plastic                  [--with_controls]
  plast_head          brake + past-block head freeze (=sweep13 winner) [--with_controls]
  derpp_plast         REPLAY + brake                             *** requested ***
  derpp_plast_head    REPLAY + brake + past-block head freeze    *** requested ***

Reading it (Pareto, per arm, mean +/- std over seeds):
  - derpp_plast vs plast, and derpp_plast_head vs plast_head: does replay ADD CIL
    and SHRINK std on top of the brake?
  - derpp_plast vs derpp: does the brake add anything on top of replay, or is
    replay already doing the work (redundant)?
  - head-freeze A/B (derpp_plast_head vs derpp_plast): head-freeze and replay BOTH
    attack the Class-IL readout wall -- freezing past-class head rows vs replaying
    old-class data to keep them calibrated. They may fight (the freeze blocks the
    replay CE from updating those rows). This tells you whether to stack them.

PROTOCOL is the standard plasticity_mlp Seq-MNIST bar (sgd, lr 0.03, batch 10,
1 epoch) -- IDENTICAL across every arm so the ONLY difference is the method. The
DER++ hyperparameters (buffer 500, alpha 1.0, beta 0.5, minibatch = batch_size)
are the mammoth Seq-MNIST buffer-500 values (scripts/osc/clser_best_args.py),
shared by every replay arm so derpp-alone and derpp+brake are apples-to-apples.

The `plast` brake center is the sweep13 WINNER (layers=fc12, accumulate, lambda
160, budget=neuron with the ramp, conductance geometry, weights_line death) --
i.e. `plast_head` here reproduces the 27.1-CIL winner, and the replay arms bolt
DER++ onto exactly that. NB: the winner center includes neuron death (recycles
weights); replay re-stabilizes replayed samples, so watch for a death<->replay
interaction (if the replay arms look pathological, the first thing to try is
--plast_death off on the brake center -- see SIMPLE_BRAKE note below).

Self-contained (no sibling imports; runs on the OSC checkout once
models/plasticity_mlp.py is synced -- the --plast_derpp* flags are new).

Usage:
    python gen_derpp_plast_sweep.py                 # 6 arms x 10 seeds = 60 runs
    python gen_derpp_plast_sweep.py --core_only      # only the 3 requested replay arms
    python gen_derpp_plast_sweep.py --buffer_sizes 200 500   # sweep buffer on the replay arms
    sbatch --array=1-$(wc -l < derpp_plast_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch derpp_plast_sweep.txt
    python aggregate_logs.py ../../logs
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Identical training protocol for EVERY arm (the whole point: only the method
# differs). --enable_other_metrics 1 turns on Forgetting/BWT/FWT in the .err log.
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')

# DER++ hyperparameters, shared by every replay arm (mammoth Seq-MNIST buffer-500
# values from clser_best_args.py: alpha 1.0, beta 0.5). Minibatch = batch_size so
# the replay draw matches the task batch. Kept identical across derpp-alone and the
# derpp+brake arms so the comparison isolates the brake.
DERPP = dict(buffer_size=500, minibatch=10, alpha=1.0, beta=0.5)

# The sweep13 WINNER brake center (layers=fc12 + past_blocks = CIL 27.1/TIL 93.2).
# `plast_head` below reproduces it; `plast` is it with head_consolidate off; the
# replay arms bolt DER++ onto it. Copied verbatim from the winning run's args.
WINNER_CENTER = dict(
    plast_signal='act_grad', plast_consistency='mean', plast_beta=0.99,
    plast_lambda=160.0, plast_lambda_end=-1.0, plast_normalize=1,
    plast_p_min=0.001, plast_p_max=20.0, plast_layers='fc12',
    plast_importance_mode='accumulate', plast_accum_floor=0.0, plast_accum_center=1,
    plast_consol_stiffness=2.0, plast_stiff_min=0.01, plast_stiff_max=1000.0,
    plast_lr_schedule='cosine', plast_lr_min=0.0, plast_warmup_epochs=0.0,
    plast_spread=1.0, plast_geometry='conductance', plast_geo_sigma=0.02,
    plast_geo_n_barriers=32, plast_geo_barrier_height=8.0, plast_geo_barrier_width=0.2,
    plast_geo_bins=256, plast_geo_every=1, plast_geo_seed=0,
    plast_death='weights_line', plast_death_thresh=0.3, plast_death_window=100,
    plast_death_warmup=500, plast_death_rate_cap=0.05, plast_death_relative=1,
    plast_death_reset_outgoing=1,
    plast_budget='neuron', plast_budget_frac=0.4, plast_budget_frac_end=0.80,
)
# SIMPLE_BRAKE note: to remove the death/geometry/budget confounds from the brake
# (a cleaner "is it the replay or the machinery" read), set on the CLI-built cfg:
#   plast_death='off', plast_geometry='none', plast_spread=0.0, plast_budget='none'
# leaving signal/lambda/accumulate. Not the default -- the default IS the winner.

# plasticity_mlp arms carry the diagnostics so aggregate_logs.py gets the
# oracle (NCM ceiling) / frozen-fraction / turnover columns.
PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 --plast_ncm_oracle 1 '
            '--plast_probe 1 --plast_geo_probe 1 --plast_turnover_probe 1')

# Arm labels in canonical per-seed emission order. Only the two REPLAY_PLAST arms
# use the new --plast_derpp* flags, so those are the ones that fail on an un-synced
# checkout -- rerun just them with `--arms derpp_plast derpp_plast_head`.
CONTROL_ARMS = ['sgd', 'plast', 'plast_head']       # no replay
REPLAY_ARMS = ['derpp', 'derpp_plast', 'derpp_plast_head']
ALL_ARMS = CONTROL_ARMS + REPLAY_ARMS


def cfg_to_flags(cfg):
    # Emit NEGATIVE values as `--flag=value` so a bare "-1.0" is never parsed as a
    # flag token (mammoth's check_multiple_defined_arg_during_string_parse treats
    # any '-'-prefixed token as a flag name; lambda_end=-1.0 would otherwise trip
    # it if a second negative flag is present).
    out = []
    for k, v in cfg.items():
        s = str(v)
        out.append(f'--{k}={s}' if s.startswith('-') else f'--{k} {s}')
    return ' '.join(out)


def derpp_line(buffer_size, seed):
    """Arm: DER++ alone (stock rehearsal, no brake). Same protocol as every arm."""
    d = DERPP
    return (f'--model derpp {PROTOCOL} '
            f'--buffer_size {buffer_size} --minibatch_size {d["minibatch"]} '
            f'--alpha {d["alpha"]} --beta {d["beta"]} --seed {seed}')


def sgd_line(seed):
    """Control: neither replay nor brake (the floor)."""
    return f'--model sgd {PROTOCOL} --seed {seed}'


def plast_line(head, derpp, buffer_size, seed):
    """A plasticity_mlp arm on the WINNER brake center.
      head  : 'off' | 'past_blocks'  (past-block head freeze)
      derpp : bool                    (bolt DER++ replay on)
    """
    cfg = dict(WINNER_CENTER)
    cfg['plast_head_consolidate'] = head
    if derpp:
        cfg['plast_derpp'] = 1
        cfg['plast_derpp_buffer_size'] = buffer_size
        cfg['plast_derpp_minibatch'] = DERPP['minibatch']
        cfg['plast_derpp_alpha'] = DERPP['alpha']
        cfg['plast_derpp_beta'] = DERPP['beta']
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def seed_lines(seed, buffer_sizes, selected):
    """(label, line) pairs for one seed, canonical order, only for arms in
    `selected`. Replay arms emit once per buffer size; controls are buffer-free."""
    out = []

    def emit(label, line):
        if label in selected:
            out.append((label, line))

    emit('sgd', sgd_line(seed))
    emit('plast', plast_line('off', False, None, seed))
    emit('plast_head', plast_line('past_blocks', False, None, seed))
    for bs in buffer_sizes:
        emit('derpp', derpp_line(bs, seed))
        emit('derpp_plast', plast_line('off', True, bs, seed))
        emit('derpp_plast_head', plast_line('past_blocks', True, bs, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--buffer_sizes', type=int, nargs='+', default=[500],
                    help='Buffer size(s) for the replay arms (each adds a run per '
                         'replay arm per seed). Default [500].')
    ap.add_argument('--arms', nargs='+', choices=ALL_ARMS, default=None,
                    metavar='ARM',
                    help='Emit only these arms (default: all 6). E.g. rerun just '
                         'the plasticity_mlp+replay combos that need the new flags: '
                         '--arms derpp_plast derpp_plast_head. Choices: '
                         + ', '.join(ALL_ARMS) + '.')
    ap.add_argument('--core_only', action='store_true',
                    help='Shorthand for --arms derpp derpp_plast derpp_plast_head '
                         '(the 3 requested replay arms; drops the controls). '
                         'Ignored if --arms is given.')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'derpp_plast_sweep.txt'))
    args = ap.parse_args()

    if args.arms:
        selected = list(dict.fromkeys(args.arms))       # de-dup, keep user order
    elif args.core_only:
        selected = list(REPLAY_ARMS)
    else:
        selected = list(ALL_ARMS)

    seen, lines = set(), []

    def add(line):
        if line not in seen:
            seen.add(line); lines.append(line)

    for seed in args.seeds:                             # seed-outer, arm-inner
        for _label, line in seed_lines(seed, args.buffer_sizes, set(selected)):
            add(line)

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"wrote {len(lines)} runs to {args.out} "
          f"(arms: {', '.join(selected)}; {len(args.seeds)} seeds; "
          f"buffer_sizes={args.buffer_sizes})")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
