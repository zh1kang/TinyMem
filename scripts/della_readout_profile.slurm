#!/bin/bash
#SBATCH --job-name=tinymem-readout-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu40
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Disposable compute profile only. No training launch or automatic follow-up.
set -eo pipefail
if ! git diff --quiet HEAD -- src scripts tests configs pyproject.toml; then
  echo "transfer committed code, not unrelated local working edits" >&2; exit 1
fi
if [[ -n $(git ls-files --others --exclude-standard -- src scripts tests configs) ]]; then
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
srun python scripts/run_readout_interface.py profile --device cuda "$@"
