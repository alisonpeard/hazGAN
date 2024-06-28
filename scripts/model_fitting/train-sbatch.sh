#!/bin/bash
#SBATCH --job-name=hazGAN_sweep
#SBATCH --output=sbatch/hazGAN_sweep_%A_%a.out
#SBATCH --error=sbatch/hazGAN_sweep_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=10:00:00

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
# RUN AGENT: (e.g., wandb agent alison-peard/hazGAN/zl1ugbxw)
wandb agent "alison-peard/hazGAN/$1"
# INITIALISE SWEEP: wandb sweep sweep.yaml
