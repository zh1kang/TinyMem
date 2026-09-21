#!/bin/bash
# Run one stage of a study runner on CPU only (writer fitting, reports).
#
#   sbatch [--array=0-N] scripts/slurm/cpu.slurm scripts/run_<study>.py <stage> [runner args]
#
# Submit from the repository root. When the job is an array, the task index is
# passed as --cell. Override any directive below on the sbatch command line.
# Set TINYMEM_ENV to a shell file that activates your Python environment when
# the default uv virtual environment (.venv) is not the one to use.
#SBATCH --job-name=tinymem-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -eo pipefail
runner="${1:?study runner script, for example scripts/run_distilled_study.py}"
shift
source "${TINYMEM_ENV:-$PWD/.venv/bin/activate}"
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache/xdg/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.cache/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" logs
python -m pip check
args=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    args+=(--cell "$SLURM_ARRAY_TASK_ID")
fi
srun python "$runner" "${args[@]}"
