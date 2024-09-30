#!/bin/bash
#SBATCH --job-name=hazGAN
#SBATCH --output=hazGAN_%A_%a.out
#SBATCH --error=hazGAN_%A_%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=12:00:00

python train.py

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ sbatch sbatch.sh

# if not working with sbatch, run
# srun -p GPU --gres=gpu:tesla:1 --time=04:00:00 --pty wandb agent alison-peard/hazGAN/8yep8d6w
# precipitation sweep: gfkbsk3l
