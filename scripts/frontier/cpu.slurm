#!/bin/bash
# Write the frontier report from a study directory whose cells are all sealed.
#
#   cd <study> && sbatch cpu.slurm
#
# Required environment: TINYMEM_ENV (shell file that activates the Python
# environment). Override any directive below on the sbatch command line.
#SBATCH --job-name=tinymem-frontier-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
source "${TINYMEM_ENV:?shell file that activates the Python environment}"
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/source/src"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p logs
srun python report.py --study "$PWD"
