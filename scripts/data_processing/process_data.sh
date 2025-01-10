#!/bin/bash
#SBATCH --job-name=data
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch/%A.out
#SBATCH --error=sbatch/%A.err
#SBATCH --partition=Medium

# conda activate hazGAN-torch # do before submitting

# -make sure RES set for all these-
# python resample_era5.sh
# python process_resampled.py
Rscript marginals.R
python make_training.py
# python make_pretraining.py

