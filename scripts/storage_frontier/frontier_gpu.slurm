#!/bin/bash
#SBATCH --job-name=tinymem-storage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --constraint=gpu40
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -eo pipefail
module purge
module load anaconda3/2026.7
conda activate tinymem-py311
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/src"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$PWD/.cache/triton/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$PWD/.cache/xdg/${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$PWD/.cache/cuda/${SLURM_JOB_ID}"
export HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$PWD/.cache/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
python -m pip check
args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--cell "$SLURM_ARRAY_TASK_ID")
fi
srun python frontier_run.py "${args[@]}" --study "$PWD" --model /scratch/gpfs/JORDANAT/caleb/TinyMem/data/raw/pretrained/qwen3-1.7b
