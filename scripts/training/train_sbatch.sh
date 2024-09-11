#!/bin/bash
#SBATCH --job-name=hazGAN
#SBATCH --output=sbatch/hazGAN_%A_%a.out
#SBATCH --error=sbatch/hazGAN_%A_%a.err
#SBATCH --array=1-100%1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3080ti:1
#SBATCH --time=10:00:00


echo "TASK ID: " $SLURM_ARRAY_TASK_ID
wandb agent alison-peard/hazGAN/$1

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch train_sbatch.sh <sweep id>

# if not working with sbatch, run
# srun -p GPU --gres=gpu:tesla:1 --time=04:00:00 --pty wandb agent alison-peard/hazGAN/bodaevqp
# GPUs: tesla, 3080ti, 1080ti
# latest sweep: s9k8wgnb
# e.g., sbatch train_sbatch.sh s9k8wgnb
