# Hazard GAN
Wasserstein GAN to generate multivariate windstorm footprints over the Bay of Bengal.
![fig2](figures/training_footprints.png)

*This branch is for developing custom torch models to use as an alternative to StyleGAN.*

## Installation
```bash
git clone git@github.com:alisonpeard/hazGAN.git
cd environments
mamba create -f apple-silicon.yaml # or other OS env
mamba activate hazGAN
python helloworld_tensorflow.py --device GPU # confirm tensorflow working
cd ..
python -m pip install -e .

pytest tests/ -x
python scripts/training/train.py --dry-run
```
