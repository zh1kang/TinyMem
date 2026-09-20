#!/bin/bash
#SBATCH --job-name=tinymem-babi-reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --constraint=gpu40
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -eo pipefail
module purge
module load "${TINYMEM_CONDA_MODULE:-anaconda3/2026.7}"
conda activate "${TINYMEM_CONDA_ENV:-tinymem-py311}"
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$PWD/.cache/triton/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$PWD/.cache/xdg/${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$PWD/.cache/cuda/${SLURM_JOB_ID}"
export HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$PWD/.cache/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
python -m pip check
srun python -m scripts.run_babi_memory_readout train "$@" --cell "${SLURM_ARRAY_TASK_ID:?array index required}"
srun python -m scripts.run_babi_memory_readout evaluate "$@" --cell "$SLURM_ARRAY_TASK_ID"
