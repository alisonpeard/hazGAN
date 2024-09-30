#!/bin/bash
#SBATCH --job-name=ncmwf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=ncmwf.out
#SBATCH --error=ncmwf.err
#SBATCH --partition=Short
#SBATCH --time=10:00:00

python request_imdaa_cluster.py

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch train-sbatch.sh <sweep id>