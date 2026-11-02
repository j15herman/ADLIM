#!/bin/bash
#SBATCH --job-name=mammoth-cl
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Same generic Mammoth runner as run_mammoth.sbatch, retargeted to OSC's
# Cardinal cluster (H100) to dodge Ascend's queue. OSC clusters share home /
# project / scratch filesystems, modules, and the conda env, so ONLY the
# resource header + login node change — the data, env, and python command
# below are byte-identical to the Ascend script.
#
#   *** Submit from the Cardinal login node ***  ssh cardinal.osc.edu
#   (OSC clusters are separate Slurm instances; sbatch routes to whichever
#    cluster's login node you submit from.)
#
#   sbatch -J cv run_mammoth_cardinal.sbatch consolidation-vit [args...]
#
# VERIFY two cluster-specific things on Cardinal before a big batch (its
# software stack is newer than Ascend's):
#   1. module name:  `module spider miniconda3`  — the pinned version below
#      may differ on Cardinal. Override the `module load` line if needed.
#   2. GPU request:  `--gpus-per-node=1` is portable, but if a job pends
#      forever or rejects, check `sinfo`/OSC Cardinal docs for a required
#      partition or typed GPU (e.g. add `--partition=...` / `--gres=gpu:h100:1`).
# Anything here can also be overridden at submit time, e.g.:
#   sbatch --partition=gpu --time=12:00:00 run_mammoth_cardinal.sbatch consolidation-vit ...
#
# H100 note: Cardinal's H100s run bf16 fast, but the reference runs are fp32
# — keep fp32 (do NOT pass --consol_bf16 1) if you want results comparable to
# the Ascend ablation; enable it only for throughput on fresh experiments.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

# VERIFY on Cardinal: `module spider miniconda3` (newer stack may rename this).
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MODEL="$1"; shift

# Shared OSC scratch — same path as Ascend, already seeded, no re-download.
DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}
SEED=${SEED:-0}

if [ ! -d "${DATA_ROOT}/CIFAR100/cifar-100-python" ]; then
    echo "ERROR: ${DATA_ROOT}/CIFAR100 not seeded — run prepare_data.sh on a login node first." >&2
    exit 1
fi

python main.py \
    --model "${MODEL}" \
    --dataset seq-cifar100-224 \
    --transform_type weak \
    --backbone vit-s-16 \
    --n_epochs 20 \
    --batch_size 64 \
    --optimizer adamw \
    --lr 1e-4 \
    --optim_wd 5e-4 \
    --lr_scheduler cosine \
    --num_workers 8 \
    --seed "${SEED}" \
    --base_path "${DATA_ROOT}/" \
    --results_path results/ \
    --checkpoint_path checkpoints/ \
    --savecheck last \
    --enable_other_metrics 1 \
    --non_verbose 1 \
    "$@"
