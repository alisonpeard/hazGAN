#!/bin/bash
#SBATCH --job-name=era5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=era5-request.out
#SBATCH --error=era5-request.err
#SBATCH --partition=Short
#SBATCH --time=10:00:00

python request_era5_cluster.py

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ wandb sweep sweep.yaml
# $ sbatch train-sbatch.sh <sweep id>