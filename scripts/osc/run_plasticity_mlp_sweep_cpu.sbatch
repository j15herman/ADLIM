#!/bin/bash
#SBATCH --job-name=plast-mlp-cpu
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

# CPU-only twin of run_plasticity_mlp_sweep.sbatch. Use on clusters whose GPUs
# this torch build does not support (e.g. Pitzer V100/V100S, CC 7.0 ->
# cudaErrorNoKernelImageForDevice). NO GPU is requested, so the job routes to the
# CPU QOS -- requesting --gpus-per-node=0 on the GPU script instead trips
# QOSMinGRES (the GPU QOS mandates >=1 GPU). These small MLP jobs run fine on CPU
# (minutes, not the ~30s of the GPU path); results are identical (device-agnostic).
#
#   python gen_head_consol_xdataset_sweep.py --with_baselines
#   sbatch --array=1-$(wc -l < head_consol_seq_fmnist_sweep.txt)%24 \
#       run_plasticity_mlp_sweep_cpu.sbatch head_consol_seq_fmnist_sweep.txt

set -euo pipefail
RUNS_FILE="${1:?usage: sbatch --array=1-N run_plasticity_mlp_sweep_cpu.sbatch runs.txt}"
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "scripts/osc/${RUNS_FILE}")
[ -n "${LINE}" ] || { echo "No line ${SLURM_ARRAY_TASK_ID} in ${RUNS_FILE}"; exit 1; }

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}

# Fail fast if this run's dataset isn't pre-seeded (concurrent compute-node
# downloads race/corrupt the archives). Derive the torchvision data dir from the
# --dataset in the run line so seq-fmnist is guarded on FashionMNIST, not MNIST.
DATASET=$(printf '%s\n' "${LINE}" | grep -oE -- '--dataset[ =][^ ]+' | head -1 | sed -E 's/--dataset[ =]//')
case "${DATASET}" in
    seq-mnist)  NEED_DIR="MNIST" ;;
    seq-fmnist) NEED_DIR="FashionMNIST" ;;
    *)          NEED_DIR="" ;;   # unknown dataset: skip the guard
esac
if [ -n "${NEED_DIR}" ] && [ ! -d "${DATA_ROOT}/${NEED_DIR}" ]; then
    echo "ERROR: ${DATA_ROOT}/${NEED_DIR} not seeded (dataset ${DATASET}) — run prepare_data.sh on a login node first." >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Run:  ${LINE}"

# shellcheck disable=SC2086
python main.py ${LINE} \
    --base_path "${DATA_ROOT}/" \
    --results_path results/ \
    --num_workers 8 \
    --non_verbose 1 \
    --device cpu
