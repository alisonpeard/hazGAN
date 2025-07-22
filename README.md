# Hazard GAN

This repo contains all code to reproduce the figures in the preprint "Simulating mulitvariate hazards with generative deep learning". Note that due to stochasticity in the StyleGAN training, some results may vary slightly.


## Installation

```bash
git clone git@github.com:alisonpeard/hazGAN.git
cd environments
mamba create -f apple-silicon.yaml # or yaml for other OS env
mamba activate hazGANv0
```

The path to the data folder `hazGAN-data` is set in the `.env` file in the parent directory. Change it to point to the location of the downloaded `hazGAN-data` folder.

This codebase is currently being migrated to a snakemake workflow to make it easier to run and modify. Some settings in the current setup are quite hardcoded.

The StyleGAN model is included in the `styleGAN-DA` folder. This model requires an nVIDIA GPU to run. The environment definition in `environments/linux-gpu.yaml` creates an environment that works with the model on linux machines.


## Contents

- Marginal distribution fits: scripts/01_data_processing/04_make_training.py
- Generated samples: scripts/02_training_inference/02_visualise.py
- Mangrove damage probability surface: scripts/03_mangrove_study/00_train.py
- Mangrove risk curve: scripts/03_mangrove_study/02_mangrove_intersect.py
