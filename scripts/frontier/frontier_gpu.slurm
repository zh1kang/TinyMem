#!/bin/bash
# Run one frontier stage inside a prepared study directory on a single GPU.
#
#   cd <study> && sbatch [--array=0-11] frontier_gpu.slurm <features|preflight|train|evaluate|transfer>
#
# The study directory carries its own source snapshot (source/src) and copies
# of the frontier_*.py scripts; this file is hashed into the protocol with them.
# Required environment: TINYMEM_ENV (shell file that activates the Python
# environment) and TINYMEM_MODEL (directory with the verified Qwen3-1.7B
# snapshot). Override any directive below on the sbatch command line.
#SBATCH --job-name=tinymem-frontier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -eo pipefail
source "${TINYMEM_ENV:?shell file that activates the Python environment}"
model="${TINYMEM_MODEL:?directory with the verified Qwen3-1.7B snapshot}"
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/src"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache/xdg/${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$PWD/.cache/cuda/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$PWD/.cache/triton/${SLURM_JOB_ID}"
export TMPDIR="$PWD/.cache/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" logs
python -m pip check
args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--cell "$SLURM_ARRAY_TASK_ID")
fi
srun python frontier_run.py "${args[@]}" --study "$PWD" --model "$model"
