#!/bin/bash
#SBATCH --job-name=plast-mlp
#SBATCH --account=PAS3236
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Single training run of the per-weight metaplasticity model (plasticity_mlp)
# on Seq-MNIST, at the EXACT CLS-ER-table Seq-MNIST sgd protocol (SGD, lr 0.03,
# batch 10, 1 epoch) so the result is apples-to-apples with the table baselines.
# Modeled on run_consolidation_mlp_cardinal.sbatch (Cardinal H100, low queue);
# the MLP is tiny so a GPU is optional — it runs fine on CPU too, the header
# just dodges Ascend's queue. OSC clusters share filesystem / modules / env, so
# only the resource header + login node differ from the Ascend scripts.
#
#   *** Submit from the Cardinal login node ***  ssh cardinal.osc.edu
#
#   sbatch run_plasticity_mlp_cardinal.sbatch
#   # override any plasticity knob at the end, e.g.:
#   sbatch run_plasticity_mlp_cardinal.sbatch --plast_lambda 3.0 --plast_consistency mean
#   SEED=1 sbatch run_plasticity_mlp_cardinal.sbatch
#
# Same-family baselines (SI, MAS) and the naive sgd floor run through
# gen_plasticity_mlp_sweep.py + run_plasticity_mlp_sweep.sbatch — don't pass the
# --plast_* flags to those models, only plasticity-mlp defines them.
#
# Diagnostics are ON (--plast_ncm_oracle 1 --plast_probe 1): the oracle logs
# head-independent Class-IL/Task-IL per task boundary, the probe logs the
# plasticity histogram (mean/std, frozen fraction, plastic fraction). Watch
# these as much as the headline accuracy — loss of plasticity shows up as the
# frozen fraction climbing toward 1 and task-5 train loss failing to fall
# (handoff §5).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/../.."   # -> mammoth repo root
mkdir -p logs results checkpoints

# VERIFY on Cardinal: `module spider miniconda3` (newer stack may rename this).
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-consolidation-mammoth}"

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DATA_ROOT=${DATA_ROOT:-/fs/scratch/PAS3236/mammoth_data}
SEED=${SEED:-0}

if [ ! -d "${DATA_ROOT}/MNIST" ]; then
    echo "ERROR: ${DATA_ROOT}/MNIST not seeded — run prepare_data.sh on a login node first." >&2
    exit 1
fi

# Plasticity defaults are spelled out (they match the model's get_parser
# defaults) so a single run is self-documenting and reproducible. Any of them
# can be overridden by appending flags at submit time ("$@").
python main.py \
    --model plasticity-mlp \
    --dataset seq-mnist \
    --backbone mnistmlp \
    --optimizer sgd \
    --lr 0.03 \
    --batch_size 10 \
    --n_epochs 1 \
    --seed "${SEED}" \
    --base_path "${DATA_ROOT}/" \
    --results_path results/ \
    --num_workers 8 \
    --enable_other_metrics 1 \
    --non_verbose 1 \
    --plast_mlp 1 \
    --plast_signal act_grad \
    --plast_consistency mean \
    --plast_beta 0.99 \
    --plast_lambda 20.0 \
    --plast_normalize 1 \
    --plast_p_min 0.0 \
    --plast_p_max 20.0 \
    --plast_layers fc12 \
    --plast_importance_mode accumulate \
    --plast_accum_floor 1.0 \
    --plast_warmup_epochs 0.0 \
    --plast_ncm_oracle 1 \
    --plast_probe 1 \
    "$@"
