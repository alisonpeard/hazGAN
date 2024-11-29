#!/bin/bash
#SBATCH --job-name=trainGAN
#SBATCH --output=sbatch/train.out
#SBATCH --error=sbatch/train.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3080ti:1
#SBATCH --time=12:00:00

python train.py