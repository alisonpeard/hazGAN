#!/bin/bash
#SBATCH --job-name=hazGAN_sweep
#SBATCH --output=sbatch/hazGAN_sweep_%A_%a.out
#SBATCH --error=sbatch/hazGAN_sweep_%A_%a.err
#SBATCH --array=1-2
#SBATCH --time=1:00:00

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
# RUN AGENT: wandb agent alison-peard/hazGAN/zl1ugbxw

# INITIALISE SWEEP: wandb sweep sweep.yaml