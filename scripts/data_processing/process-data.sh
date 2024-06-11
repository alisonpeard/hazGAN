#!/bin/zsh

source activate hazGAN

# python resample-era5.py
python b_process_raw_era5.py
Rscript c_fit_gpd.R
python d_make_training_era5.py