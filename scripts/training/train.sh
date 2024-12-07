#!/bin/bash
#SBATCH --job-name=trainGAN
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3080ti:1
#SBATCH --time=16:00:00

python train.py
