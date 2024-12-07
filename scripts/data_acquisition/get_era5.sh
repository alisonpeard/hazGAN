#!/bin/bash
#SBATCH --job-name=era5_1940s
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_dump/era5_%A_%a.out
#SBATCH --error=sbatch_dump/era5_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=Short
#SBATCH --time=06:00:00

python request_era5_yearly__soge.py $SLURM_ARRAY_TASK_ID
