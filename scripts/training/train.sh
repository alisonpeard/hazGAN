#!/bin/bash
#SBATCH --job-name=trainGAN
#SBATCH --output=sbatch/train.out
#SBATCH --error=sbatch/train.err
#SBATCH --nodelist ouce-cn19
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3080ti:1
#SBATCH --time=16:00:00

python train.py
