#!/bin/zsh

source activate hazGAN
python resample_era5.py -r 22 18
python process_resampled.py
Rscript extreme_value_marginals.R
python make_training.py