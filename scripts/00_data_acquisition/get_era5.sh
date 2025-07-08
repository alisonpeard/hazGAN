#!/bin/bash
#SBATCH --job-name=era5_1940s
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_dump/era5_%A_%a.out
#SBATCH --error=sbatch_dump/era5_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=Short
#SBATCH --time=00:30:00

python get_era5.py $SLURM_ARRAY_TASK_ID
