#!/bin/bash
#SBATCH --job-name=styleGAN2
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=GPU
#SBATCH --time=1-00:00:00

DATADIR=/soge-home/projects/mistral/alison/data/stylegan_output
source /lustre/soge1/users/spet5107/micromamba/etc/profile.d/micromamba.sh

micromamba activate styleGAN
DATADIR=/soge-home/projects/mistral/alison/data/stylegan
python ../../styleGAN-DA/src/train.py --data=${DATADIR}/images.zip --outdir=${DATADIR}/training-runs --gpus=2 --DiffAugment=color,translation,cutout --kimg=50
