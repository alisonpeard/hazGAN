conda activate hazGAN

python resample_era5.sh
Rscript marginals.R
python make_training.py
python make_pretraining.py

