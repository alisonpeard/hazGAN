#!/bin/bash
#SBATCH --job-name=era5_request
#SBATCH --output=era5_request_%A_%a.out
#SBATCH --error=era5_request_%A_%a.err
#SBATCH --array=1-876
#SBATCH --time=01:00:00

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
python era5_api_request.py $SLURM_ARRAY_TASK_ID
