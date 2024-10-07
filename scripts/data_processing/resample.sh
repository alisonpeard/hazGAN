#!/bin/bash
#SBATCH --job-name=resample-era5
#SBATCH --array=1-82%10
#SBATCH --partition=Short
#SBATCH --time=00:10:00
#SBATCH --output=resample-era5-%A_%a.out
#SBATCH --error=resample-era5-%A_%a.err

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
python resample_era5.py -y $SLURM_ARRAY_TASK_ID -r 28 28  

# To run:
# -------
# micromamba activate hazGAN
# sbatch resample.sh