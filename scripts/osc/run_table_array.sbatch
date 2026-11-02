#!/bin/bash
#SBATCH --job-name=clser-table
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

# Array runner for the CLS-ER Table-1 replication grid.
#
#   python gen_table_runs.py --seeds 0 1 2
#   sbatch --array=1-$(wc -l < runs.txt)%24 run_table_array.sbatch runs.txt
#
# Each array task runs one line of runs.txt. Most MNIST jobs finish in
# minutes; S-CIFAR-10 (50 epochs) and S-Tiny-ImageNet (100 epochs, GSS
# notably slow) need the long limit.

set -euo pipefail
RUNS_FILE="${1:?usage: sbatch --array=1-N run_table_array.sbatch runs.txt}"
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "scripts/osc/${RUNS_FILE}")
[ -n "${LINE}" ] || { echo "No line ${SLURM_ARRAY_TASK_ID} in ${RUNS_FILE}"; exit 1; }

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}

# Fail fast if datasets aren't pre-seeded (see prepare_data.sh): compute-node
# downloads stall/corrupt when many array tasks race for the same files.
if [ ! -d "${DATA_ROOT}/MNIST" ] || [ ! -d "${DATA_ROOT}/CIFAR10/cifar-10-batches-py" ]; then
    echo "ERROR: ${DATA_ROOT} not seeded — run prepare_data.sh on a login node first." >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Run:  ${LINE}"

# shellcheck disable=SC2086
python main.py ${LINE} \
    --base_path "${DATA_ROOT}/" \
    --results_path results/ \
    --num_workers 8 \
    --non_verbose 1
