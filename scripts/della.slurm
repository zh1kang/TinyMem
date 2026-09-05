#!/bin/bash
#SBATCH --job-name=tinymem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu40
#SBATCH --time=00:15:00
#SBATCH --chdir=/scratch/gpfs/JORDANAT/caleb/TinyMem
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
if ! git diff --quiet HEAD -- src scripts tests pyproject.toml; then
  echo "source changes are not committed; resolve them before running the frozen study" >&2
  exit 1
fi
module purge
module load "${TINYMEM_CONDA_MODULE:-anaconda3/2026.7}"
conda activate "${TINYMEM_CONDA_ENV:-tinymem-py311}"
set -u

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/gpfs/JORDANAT/caleb/.cache/huggingface
export PYTHONUNBUFFERED=1

nvidia-smi
python -m pip check
srun bash scripts/della_run.sh "$@"
