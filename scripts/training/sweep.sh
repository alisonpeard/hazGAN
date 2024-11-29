#!/bin/bash
#SBATCH --job-name=sweepGAN
#SBATCH --output=sbatch/sweep_%A_%a.out
#SBATCH --error=sbatch/sweep_%A_%a.err
#SBATCH --array=1-20%1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3080ti:1
#SBATCH --time=12:00:00


echo "TASK ID: " $SLURM_ARRAY_TASK_ID
wandb agent alison-peard/hazGAN/$1

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch sweep.sh <sweep id>

# if not working with sbatch, run
# srun -p GPU --gres=gpu:tesla:1 --time=04:00:00 --pty wandb agent alison-peard/hazGAN/bodaevqp
# GPUs: tesla, 3080ti, 1080ti
# latest sweep: s9k8wgnb
# e.g., sweep.sh dslczmgj
