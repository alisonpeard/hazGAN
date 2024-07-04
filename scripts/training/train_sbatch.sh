#!/bin/bash
#SBATCH --job-name=hazGAN_sweep
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch/hazGAN_sweep_%A_%a.out
#SBATCH --error=sbatch/hazGAN_sweep_%A_%a.err
#SBATCH --array=1-50%10
#SBATCH --partition=Medium
#SBATCH --time=16:00:00

echo "TASK ID: " $SLURM_ARRAY_TASK_ID
wandb agent "alison-peard/hazGAN/$1"

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch train-sbatch.sh <sweep id>
