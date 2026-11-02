#!/bin/bash
#SBATCH --job-name=consol-mlp-sweep
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

# Array runner for the sparse-MLP consolidation sweep on Seq-MNIST. Mirrors
# run_table_array.sbatch: each array task runs one line of the runs file.
# Seq-MNIST/MLP jobs finish in minutes, so the 1h limit is generous.
#
#   python gen_consolidation_mlp_sweep.py --seeds 0
#   sbatch --array=1-$(wc -l < consol_mlp_sweep.txt)%24 \
#       run_consolidation_mlp_sweep.sbatch consol_mlp_sweep.txt
#
# Aggregate with aggregate_table.py (results land under results/ keyed by
# model/dataset), or grep the slurm logs for the "consol NCM ORACLE" /
# "consol PROBE" lines to read the diagnostics per task boundary.

set -euo pipefail
RUNS_FILE="${1:?usage: sbatch --array=1-N run_consolidation_mlp_sweep.sbatch consol_mlp_sweep.txt}"
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "scripts/osc/${RUNS_FILE}")
[ -n "${LINE}" ] || { echo "No line ${SLURM_ARRAY_TASK_ID} in ${RUNS_FILE}"; exit 1; }

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}

# Fail fast if MNIST isn't pre-seeded (compute-node downloads race/corrupt).
if [ ! -d "${DATA_ROOT}/MNIST" ]; then
    echo "ERROR: ${DATA_ROOT}/MNIST not seeded — run prepare_data.sh on a login node first." >&2
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
