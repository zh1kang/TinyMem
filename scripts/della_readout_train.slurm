#!/bin/bash
#SBATCH --job-name=tinymem-readout-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu40
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One complete arm/seed, with all schedule arguments supplied by the caller.
# Resource defaults are unprofiled requests, not measured requirements.
set -eo pipefail
if ! git diff --quiet HEAD -- src scripts tests configs pyproject.toml; then
  echo "transfer committed code, not unrelated local working edits" >&2; exit 1
fi
untracked=$(git ls-files --others --exclude-standard -- src scripts tests configs)
if [[ -n "$untracked" ]]; then
  echo "untracked execution sources must be reviewed before submission" >&2; exit 1
fi
module purge
module load "${TINYMEM_CONDA_MODULE:-anaconda3/2026.7}"
conda activate "${TINYMEM_CONDA_ENV:-tinymem-py311}"
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONUNBUFFERED=1
nvidia-smi
python -m pip check
srun python scripts/run_readout_interface.py run --device cuda "$@"
