#!/bin/bash
#SBATCH --job-name=styleGAN2
#SBATCH --output=./train.out
#SBATCH --error=./train.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source /lustre/soge1/users/spet5107/micromamba/etc/profile.d/micromamba.sh
micromamba activate hazGAN

DATADIR=/soge-home/projects/mistral/alison/data/stylegan
python ../../styleGAN-DA/src/train.py --data=${DATADIR}/images.zip --outdir=${DATADIR}/training-runs --gpus=1 --DiffAugment=color,translation,cutout --kimg=50
