#!/bin/bash
#SBATCH --job-name=hazGAN
#SBATCH --output=sbatch/hazGAN_%A_%a.out
#SBATCH --error=sbatch/hazGAN_%A_%a.err
#SBATCH --array=1-100%5
#SBATCH --partition=GPU
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=03:00:00


echo "TASK ID: " $SLURM_ARRAY_TASK_ID
wandb agent alison-peard/hazGAN/$1

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch train_sbatch.sh <sweep id>

# if not working with sbatch, run
# srun -p GPU --gres=gpu:tesla:1 --time=04:00:00 --pty wandb agent alison-peard/hazGAN/8yep8d6w
# precipitation sweep: gfkbsk3l
