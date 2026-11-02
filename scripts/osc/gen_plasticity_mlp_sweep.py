"""
Generate a sweep grid for the per-weight metaplasticity model (plasticity_mlp)
on Seq-MNIST.

One-factor-at-a-time (OFAT) around a center config: each plasticity knob is
varied in turn while the others stay at the center, plus a few A/B toggles
(signal, consistency, layers). First-pass sweep — cheap, readable, enough to
find the sensitive axes before a full grid / multi-seed run (default one seed).

Every plasticity line runs the diagnostics (NCM oracle + plasticity histogram)
so each config is judged on the head-independent representation metric and the
freezing/loss-of-plasticity picture, not just headline accuracy. The
head-to-head baselines (HANDOFF_plasticity_mlp.md §3) are emitted alongside at
the IDENTICAL Seq-MNIST protocol: sgd (naive floor), and the same-family
importance baselines SI and MAS (hyperparameters from clser_best_args.py; MAS
e_lambda is also swept since no upstream-tuned value exists). A plasticity_mlp
OFF row is included as the byte-identical-to-sgd sanity check.

Usage:
    python gen_plasticity_mlp_sweep.py [--seeds 0] [--out plast_mlp_sweep.txt]
Then:
    sbatch --array=1-$(wc -l < plast_mlp_sweep.txt)%24 \
        run_plasticity_mlp_sweep.sbatch plast_mlp_sweep.txt
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Seq-MNIST sgd protocol — identical to the CLS-ER table runs.
PROTOCOL = ('--dataset seq-mnist --backbone mnistmlp --optimizer sgd '
            '--lr 0.03 --batch_size 10 --n_epochs 1 --enable_other_metrics 1')
PLAST_ON = ('--model plasticity-mlp --plast_mlp 1 '
            '--plast_ncm_oracle 1 --plast_probe 1')

# Center config for the OFAT sweep (the proven OSC default: mean-calibrated
# Omega brake, no warmup, persistent accumulator + monotonic floor, lambda 20).
CENTER = dict(
    plast_signal='act_grad',
    plast_consistency='mean',        # effect-size importance retains best (sweep 1)
    plast_beta=0.99,
    plast_lambda=20.0,               # brake strength (Omega_hat mean 1)
    plast_normalize=1,               # fixed-scale Omega calibration
    plast_p_min=0.0,                 # allow full freeze
    plast_p_max=20.0,                # caps the boost branch (accumulate)
    plast_layers='fc12',
    plast_importance_mode='accumulate',  # persistent signed accumulator
    plast_accum_floor=1.0,           # MONOTONIC: never erode (anti-erosion hypothesis)
    plast_spread=0.0,                # heterosynaptic spillover OFF at the anchor
    plast_spread_axis='post',        # meanfield neuron (row) when spread>0
    plast_spread_kernel='meanfield', # meanfield (global row mean) vs coact (local)
    plast_coact_thresh=0.0,          # coact locality threshold
    plast_warmup_epochs=0.0,         # acts from step 1
)

# Each axis: list of values to try (center value included for a clean anchor).
AXES = {
    # The headline test: does flooring erosion carry the 2-task retention to 5
    # tasks? -1 = OFF (signed, erodes, sweep-1 behavior); 0/0.5/1 = bounded->monotonic.
    'plast_accum_floor': [-1.0, 0.0, 0.5, 1.0],
    'plast_importance_mode': ['ema', 'accumulate', 'hybrid'],  # recency vs persistent
    'plast_consistency': ['snr', 'mean', 'sq'],       # SNR vs effect-size/second-moment
    'plast_lambda': [3.0, 10.0, 20.0, 30.0],          # brake strength
    'plast_beta': [0.0, 0.97, 0.99, 0.999],           # 0 = no smoothing (raw batch signal)
    'plast_p_min': [0.0, 0.1, 0.3],                   # freeze floor
    'plast_layers': ['fc12', 'all'],                  # include classifier head?
    # STC heterosynaptic spillover: how much consolidation leaks to neuron-mates.
    # The hypothesis: coherent per-neuron freezing keeps the feature the head
    # reads stable, so the head-Class-IL gap (oracle ~75 vs head ~19.5) closes.
    'plast_spread': [0.0, 0.25, 0.5, 0.75, 1.0],
}

# Standalone A/B configs that change coupled knobs at once.
EXTRA_CONFIGS = [
    # Monotonic + protect the head too (readout is the Class-IL bottleneck).
    {'plast_layers': 'all', 'plast_lambda': 30.0},
    # Monotonic + second-moment importance at a stronger brake.
    {'plast_consistency': 'sq', 'plast_lambda': 30.0},
    # The eroding signed accumulator (sweep-1 default) for a clean A/B vs monotonic.
    {'plast_accum_floor': -1.0},
    # The pure-brake recency baseline (ema) for a clean contrast vs accumulate.
    {'plast_importance_mode': 'ema'},
    # STC x best-representational mode: ema preserved features best (HANDOFF §5),
    # and spillover should make that protection neuron-coherent. The core test.
    {'plast_importance_mode': 'ema', 'plast_spread': 0.5},
    {'plast_importance_mode': 'ema', 'plast_spread': 1.0},
    # Spillover axis A/B at moderate strength (post is the default; try pre/both).
    {'plast_spread': 0.5, 'plast_spread_axis': 'pre'},
    {'plast_spread': 0.5, 'plast_spread_axis': 'both'},
    # Strong neuron-level freeze on the persistent accumulator center.
    {'plast_spread': 1.0, 'plast_lambda': 10.0},
    # --- LOCAL co-activation kernel (coact, step 9) ---
    # Mean-field spreading washed out within-neuron structure (every synapse on a
    # neuron got the same importance). coact spills only between co-firing inputs
    # (synaptic clustering), keeping sub-neuron structure. This is the refinement.
    {'plast_spread': 0.5, 'plast_spread_kernel': 'coact'},
    {'plast_spread': 1.0, 'plast_spread_kernel': 'coact'},
    # coact x ema (best representational mode) -- the core test of the refinement.
    {'plast_importance_mode': 'ema', 'plast_spread': 0.5, 'plast_spread_kernel': 'coact'},
    {'plast_importance_mode': 'ema', 'plast_spread': 1.0, 'plast_spread_kernel': 'coact'},
    # Locality strength: higher thresh = tighter branches (fewer co-active pairs).
    {'plast_spread': 0.5, 'plast_spread_kernel': 'coact', 'plast_coact_thresh': 0.3},
    {'plast_spread': 0.5, 'plast_spread_kernel': 'coact', 'plast_coact_thresh': 0.6},
]


def cfg_to_flags(cfg: dict) -> str:
    return ' '.join(f'--{k} {v}' for k, v in cfg.items())


def plast_line(cfg: dict, seed: int) -> str:
    return f'{PLAST_ON} {PROTOCOL} {cfg_to_flags(cfg)} --seed {seed}'


def load_best_args():
    ns = {}
    with open(os.path.join(HERE, 'clser_best_args.py')) as f:
        exec(f.read(), ns)
    return ns['best_args']['seq-mnist']


def baseline_lines(seed: int, dataset: str = 'seq-mnist') -> list:
    """The head-to-head baselines at the same protocol on `dataset`:
    sgd (floor), plasticity_mlp OFF (sanity), SI, MAS (e_lambda swept), oEWC
    (`ewc_on`) and LwF -- the rehearsal-free regularization family.

    Hyperparameters always come from clser_best_args['seq-mnist'] (the only tuned
    mnistmlp set). For a sibling dataset like seq-fmnist -- same 5x2 protocol,
    same backbone, no dataset-specific tuning -- those values are reused verbatim
    at `dataset`, exactly as the plasticity center is reused across datasets."""
    best = load_best_args()
    protocol = PROTOCOL.replace('--dataset seq-mnist', f'--dataset {dataset}')
    lines = []
    # sgd (naive floor) — and plasticity_mlp OFF, the byte-identical sanity row.
    cfg = best['sgd'][-1]
    sgd_flags = (f"--lr {cfg['lr']} --batch_size {cfg['batch_size']} "
                 f"--n_epochs {cfg['n_epochs']}")
    lines.append(f"--model sgd --dataset {dataset} {sgd_flags} "
                 f"--enable_other_metrics 1 --seed {seed}")
    lines.append(f"--model plasticity-mlp --plast_mlp 0 {protocol} --seed {seed}")
    # SI (present, tuned).
    cfg = best['si'][-1]
    si_flags = ' '.join(f'--{k} {v}' for k, v in cfg.items())
    lines.append(f"--model si --dataset {dataset} {si_flags} "
                 f"--enable_other_metrics 1 --seed {seed}")
    # MAS (ported): sweep e_lambda since no upstream-tuned value exists.
    cfg = best['mas'][-1]
    for e_lambda in (0.1, 1.0, 10.0):
        mas = dict(cfg, e_lambda=e_lambda)
        mas_flags = ' '.join(f'--{k} {v}' for k, v in mas.items())
        lines.append(f"--model mas --dataset {dataset} {mas_flags} "
                     f"--enable_other_metrics 1 --seed {seed}")
    # oEWC (online EWC, model `ewc_on`) — tuned seq-mnist args (e_lambda 90).
    cfg = best['ewc_on'][-1]
    ewc_flags = ' '.join(f'--{k} {v}' for k, v in cfg.items())
    lines.append(f"--model ewc_on --dataset {dataset} {ewc_flags} "
                 f"--enable_other_metrics 1 --seed {seed}")
    # LwF — tuned seq-mnist args. clser_best_args names weight decay 'wd_reg'
    # (older mammoth); this checkout's lwf takes it via the optimizer's --optim_wd.
    cfg = dict(best['lwf'][-1])
    if 'wd_reg' in cfg:
        cfg['optim_wd'] = cfg.pop('wd_reg')
    lwf_flags = ' '.join(f'--{k} {v}' for k, v in cfg.items())
    lines.append(f"--model lwf --dataset {dataset} {lwf_flags} "
                 f"--enable_other_metrics 1 --seed {seed}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='seeds (default one; full seed sweep comes later)')
    ap.add_argument('--out', type=str,
                    default=os.path.join(HERE, 'plast_mlp_sweep.txt'))
    ap.add_argument('--no_baselines', action='store_true',
                    help='omit the sgd/si/mas baseline rows')
    args = ap.parse_args()

    seen = set()
    lines = []

    def add(line):
        if line not in seen:
            seen.add(line)
            lines.append(line)

    for seed in args.seeds:
        if not args.no_baselines:
            for bl in baseline_lines(seed):
                add(bl)
        # Center anchor.
        add(plast_line(dict(CENTER), seed))
        # OFAT axes.
        for axis, values in AXES.items():
            for v in values:
                cfg = dict(CENTER)
                cfg[axis] = v
                add(plast_line(cfg, seed))
        # Coupled A/B configs.
        for extra in EXTRA_CONFIGS:
            cfg = dict(CENTER)
            cfg.update(extra)
            add(plast_line(cfg, seed))

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"wrote {len(lines)} runs to {args.out}")
    print(f"submit: sbatch --array=1-{len(lines)}%24 "
          f"run_plasticity_mlp_sweep.sbatch {os.path.basename(args.out)}")


if __name__ == '__main__':
    main()
