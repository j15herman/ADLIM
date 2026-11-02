# Consolidation port (added to stock Mammoth)

Hippocampal-indexed consolidation as a Mammoth `ContinualModel`, evaluated
against standard baselines under one shared protocol: **ViT-S/16 @ 224,
from scratch, Split CIFAR-100 (10×10), 20 epochs/task, AdamW lr 1e-4 wd 5e-4,
batch 64**.

## What was added (everything else is stock Mammoth)

| File | Purpose |
|---|---|
| `models/consolidation_utils/` | The `consolidation` package, copied verbatim (bank, encoders, module, batched). |
| `models/consolidation_vit.py` | The wrapper (`--model consolidation-vit`). |
| `backbone/vit_small.py` | Registers `--backbone vit-s-16` (from-scratch by default). |
| `scripts/osc/` | OSC sbatch runner + submit-all + env setup. |

## How the wrapper works

- No split-QKV surgery. A forward hook on each block's fused `attn.qkv`
  Linear sees the block input `X` (B, N, D) and the fused output (B, N, 3D),
  sliced as `[..., :D] / [D:2D] / [2D:]` → Y_q / Y_k / Y_v (timm's fused
  layout is `[q | k | v]` along the last dim, matching weight rows).
- Weight updates bind to **row-slice views** of the fused weight
  (`qkv.weight.data[:D]`, etc.); the consolidation module's in-place
  `.data.add_()` writes the live parameter. Binding happens lazily in
  `begin_task` because `.to(device)` re-allocates parameter storage and
  would orphan views taken at construction.
- Hooks only cache during `observe()`'s forward (`self._caching`), so
  evaluation passes cost nothing. Probe/TTC scripts can call
  `model.set_caching(True)` to fill the cache during custom eval loops.
- `observe()` = forward → backward → `opt.step()` → SupCon projection step →
  `consolidate()` (same order as the bespoke `train_cl.py`; SupCon runs
  first because `consolidate()` clears the cache).
- consolidate() exceptions are counted and surfaced (`error_summary()`),
  never silently swallowed — per-task log line reports calls/errors.

Hyperparameter defaults replicate `run_consolidation_topk3.sbatch`
(alpha 5e-4, normalize_delta, 64 slots, top-k 3 @ temp 0.1, projection 128,
20-epoch bank warmup, projection trained for the whole run).

**Known deltas vs the bespoke runs** (shared by all methods, so comparisons
stay fair): constant LR instead of per-task cosine-with-warmup; fp32 instead
of bf16 autocast (`--consol_bf16 1` re-enables it for this model only).

## Run

Local smoke test (CPU, ~5 min):

```
python main.py --model consolidation-vit --dataset seq-cifar100-224 \
  --backbone vit-s-16 --lr 1e-4 --optimizer adamw --optim_wd 5e-4 \
  --n_epochs 1 --batch_size 8 --debug_mode 1 --num_workers 0 --seed 0 \
  --consol_warmup_steps 2 --consol_log_every 1
```

OSC (from `scripts/osc/`): `bash setup_env.sh` once, then
`SEED=0 bash submit_all.sh` for the full grid (consolidation, sgd, ewc_on,
lwf, er @ 200/500/5120, derpp @ 500), or individual jobs via
`sbatch -J name run_mammoth.sbatch <model> [args...]`.

Checkpoints land in `checkpoints/` (`--savecheck last`) for the probe/TTC
eval scripts; class-IL + task-IL accuracy matrices, forgetting, and BWT/FWT
come from Mammoth's standard logging (`data/results/`).

## CLS-ER Table-1 replication

`models/clser.py` is a port of the official CLS-ER (NeurAI-Lab/CLS-ER fork)
to the current Mammoth API; `forward()` runs the **stable model**, matching
the metric the paper reports. The full published grid (S-MNIST / S-CIFAR-10 /
S-Tiny-ImageNet class-IL + R-MNIST / P-MNIST domain-IL × buffers 200/500/5120
× {JOINT, SGD, ER, GEM, iCaRL, FDR, GSS, DER++, CLS-ER}) is generated from
the fork's own `best_args.py` (the source of the published numbers):

```
cd scripts/osc
python gen_table_runs.py --seeds 0 1 2          # -> runs.txt (309 jobs)
sbatch --array=1-$(wc -l < runs.txt)%24 run_table_array.sbatch runs.txt
```

The "-" cells of the paper (GEM/GSS on Tiny-ImageNet, iCaRL on Domain-IL) are
skipped automatically. Renames handled: icarl `wd_reg`→`opt_wd`; old
`softmax_temp` no longer exists in modern icarl (implementation drifted —
expect small deviations on iCaRL rows). JOINT uses SGD's per-dataset
lr/batch/epochs (best_args has no JOINT entry). GEM needs `quadprog`
(in setup_env.sh; installs fine on modern pips despite the in-repo warning).

## Caveats for the writeup

- **Memory accounting**: the default bank (64 slots × 12 layers, keys +
  3×per-head values + projections) stores ~230M floats — ~15× the floats in
  Mammoth's largest standard buffer (5120 CIFAR images ≈ 15.7M). Either
  report a memory-matched comparison against er5120 honestly, or shrink the
  bank (fewer layers / fp16 / fewer slots) toward a standard budget.
- Baseline method hyperparameters in `submit_all.sh` (DER++ α/β, EWC λ) are
  Mammoth's usual values, not tuned for AdamW+ViT-from-scratch — sweep
  before quoting them.
- CLS-ER is not in Mammoth master; the official CLS-ER repo is itself a
  Mammoth fork (NeurAI-Lab/CLS-ER) — its `models/clser.py` should drop in.
- timm's CIFAR-100 download URL may 403; pre-seed
  `<base_path>/CIFAR100/cifar-100-python.tar.gz`
  (md5 `eb9058c3a382ffc7106e4002c42a8d85`).
