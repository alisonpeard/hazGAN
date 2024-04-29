#! /bin/zsh

source conda activate general
python "1a_process_era5.py"
R -f "1b_fit_marginals_era5.R"
python "1c_make_training_era5.py"
source activate tf_geo
python "train_wgan.py"