#!/bin/bash
#SBATCH --job-name=data
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch/%A.out
#SBATCH --error=sbatch/%A.err
#SBATCH --partition=Medium

# source activate micromamba
# micromamba activate hazGAN-torch # do before submitting

# -make sure RES set for all these-
python resample_slurm.sh
python process_resampled.py

Rscript extract_storms.R
Rscript fit_marginals.R

python make_training.py
python make_jpegs.py

