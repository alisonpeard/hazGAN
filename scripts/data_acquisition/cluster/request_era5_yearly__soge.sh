#!/bin/bash
#SBATCH --job-name=era5
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=era5.out
#SBATCH --error=era5.err
#SBATCH --array=1-83%10
#SBATCH --partition=Short
#SBATCH --time=06:00:00

python request_era5_from_soge.py $SLURM_ARRAY_TASK_ID
