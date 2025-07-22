#!/bin/bash
#SBATCH --job-name=resample
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch/%A_%a.out
#SBATCH --error=sbatch/%A_%a.err
#SBATCH --array=1-82
#SBATCH --partition=Short
#SBATCH --time=00:30:00

YEARS=($(seq 1940 2022))

python resample.py --year ${YEARS[$SLURM_ARRAY_TASK_ID]} --res 64 64

