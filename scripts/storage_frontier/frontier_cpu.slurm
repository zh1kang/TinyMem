#!/bin/bash
#SBATCH --job-name=tinymem-storage-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
module purge
module load anaconda3/2026.7
conda activate tinymem-py311
set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD/source/src"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
srun python frontier_report.py --study "$PWD"
