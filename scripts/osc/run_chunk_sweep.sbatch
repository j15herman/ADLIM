#!/bin/bash
#SBATCH --job-name=cv-chunksweep
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

# Chunk-granularity sweep for consolidation-vit, as a single array job.
# Per-layer path, sequence padded to 256 so composite chunk counts that the
# prime backbone length (197) forbids become reachable.
#
#   sbatch --array=0-3 run_chunk_sweep.sbatch
#
# Array index -> chunk_length -> n_chunks (max_seq_length = 256):
#   0:  256 -> 1     1:  64 -> 4     2:  16 -> 16     3:  4 -> 64
#
# Everything else matches the bespoke protocol: ViT-S/16 @ 224 from scratch,
# Split CIFAR-100 (10x10), 20 epochs/task, AdamW lr 1e-4 wd 5e-4, batch 64,
# cosine LR, weak aug. Only chunk granularity varies across tasks.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

CHUNK_LENGTHS=(256 64 16 4)
CHUNK_LENGTH=${CHUNK_LENGTHS[${SLURM_ARRAY_TASK_ID:?must submit with --array=0-3}]}
MAX_SEQ=256
N_CHUNKS=$(( MAX_SEQ / CHUNK_LENGTH ))

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}
SEED=${SEED:-0}

# Fail fast if the dataset isn't pre-seeded: compute-node downloads stall
# against cs.toronto.edu and racing array tasks corrupt each other's files.
if [ ! -d "${DATA_ROOT}/CIFAR100/cifar-100-python" ]; then
    echo "ERROR: ${DATA_ROOT}/CIFAR100 not seeded — run prepare_data.sh on a login node first." >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Sweep task ${SLURM_ARRAY_TASK_ID}: chunk_length=${CHUNK_LENGTH} -> ${N_CHUNKS} chunks (max_seq=${MAX_SEQ})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python main.py \
    --model consolidation-vit \
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
    --consol_top_k 1 \
    --consol_data_init 1 \
    --consol_dead_restart_interval 200 \
    --consol_use_batched 0 \
    --consol_max_seq_length "${MAX_SEQ}" \
    --consol_chunk_length "${CHUNK_LENGTH}"
