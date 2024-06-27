#!/bin/bash
#SBATCH --job-name=sweep_agent
#SBATCH --output=era5_request_%A_%a.out
#SBATCH --error=era5_request_%A_%a.err
#SBATCH --array=1-50
#SBATCH --time=10:00:00

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
# RUN AGENT: wandb agent alison-peard/hazGAN/zl1ugbxw

# INITIALISE SWEEP: wandb sweep sweep.yaml