#!/bin/bash
#SBATCH --job-name=mammoth-cl
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Generic Mammoth runner for the consolidation-vs-baselines comparison.
#
#   sbatch run_mammoth.sbatch <model> [extra mammoth args...]
#
# e.g.
#   sbatch -J cv      run_mammoth.sbatch consolidation-vit
#   sbatch -J sgd     run_mammoth.sbatch sgd
#   sbatch -J er500   run_mammoth.sbatch er    --buffer_size 500 --minibatch_size 64
#   sbatch -J derpp   run_mammoth.sbatch derpp --buffer_size 500 --minibatch_size 64 --alpha 0.1 --beta 0.5
#   sbatch -J ewc     run_mammoth.sbatch ewc_on --e_lambda 10 --gamma 1.0
#   sbatch -J lwf     run_mammoth.sbatch lwf
#
# All methods share the exact backbone/protocol of the bespoke runs:
# ViT-S/16 @ 224 from scratch, Split CIFAR-100 (10x10), 20 epochs/task,
# AdamW lr 1e-4 wd 5e-4, batch 64, cosine LR schedule, weak augmentation.
# This is the whole point of the port:
# baselines under identical conditions, with Mammoth's standard class-IL
# and task-IL accuracy matrices, forgetting, and BWT/FWT for free.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MODEL="$1"; shift

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}
SEED=${SEED:-0}

# Fail fast if the dataset isn't pre-seeded: compute-node downloads stall
# against cs.toronto.edu and concurrent jobs corrupt each other's files.
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
