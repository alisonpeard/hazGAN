#!/bin/zsh

python b_process_raw_era5.py
Rscript c_fit_gpd_era5.R
python d_make_training_era5.py