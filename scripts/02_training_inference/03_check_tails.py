"""
Compare generated and training data tails for different training
configurations.
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from environs import Env

# script parameters
DOMAINS   = ["uniform", "gaussian", "gumbel", "rescaled"]
VERSION   = ["", "-04", "-05", "-06"][0]
FIELD     = [0, 1, 2][0]
THRESH    = [0.0, 0.8][0]
ORIENT    = ["portrait", "landscape"][0]
# set font to monospace for better readability
plt.rcParams.update({'font.family': 'monospace'})
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 7, 'axes.titlesize': 7, 'legend.fontsize': 6})

def load_pngs(png_dir):
    png_list = os.listdir(png_dir)
    png_list = [png for png in png_list if not png.startswith(".")]
    png_list = [os.path.join(png_dir, png) for png in png_list]
    png_list = sorted(png_list)

    samples = []
    for png in (pbar := tqdm(png_list, desc=f"Loading files from {png_dir}")):
        pbar.set_description(f"Loading {os.path.basename(png)}")
        with Image.open(png) as img:
            samples.append(np.array(img))
    samples = np.array(samples).astype(float)
    samples /= 255.
    print(f"Loaded {samples.shape} samples")
    return samples


if __name__ == "__main__":
    
    env = Env()
    env.read_env()

    if ORIENT == "portrait":
        fig, axes = plt.subplots(4, 2, figsize=(3.35, 6), sharex='col', sharey='col')

        for j, THRESH in enumerate([0.0, 0.8]):
            bins = np.linspace(THRESH, 1.0, 20)
            hist_kws = dict(density=True, bins=bins, alpha=0.6, edgecolor='k', linewidth=0.4, histtype='stepfilled')

            for i, DOMAIN in enumerate(DOMAINS):
                TRAIN   = os.path.join(env.str("TRAINDIR"), "images", DOMAIN, "storm")
                SAMPLES = os.path.join(env.str("SAMPLES_DIR"), f"{DOMAIN}{VERSION}/gen")
                train = load_pngs(TRAIN)
                gen   = load_pngs(SAMPLES)
                headroom = gen.max() - train.max()
                print(f"Headroom between training and generated samples for {DOMAIN}: {headroom}")
                
                train = train[..., FIELD].ravel()
                train = train[train > THRESH]

                gen = gen[..., FIELD].ravel()
                gen = gen[gen > THRESH]

                ax = axes[i, j]
        
                ax.hist(gen, label="HazGAN", **hist_kws, color="#002147");
                ax.hist(train, label="ERA5", **hist_kws, color="#C8D1DF");
                if i == j == 0:
                    ax.legend(loc="best", frameon=False)
                if i == 0:
                    ax.set_title(f"y ≥ {THRESH:.1f}")
                if j == 0:
                    ax.set_ylabel(f"{DOMAIN.capitalize()}\ndensity", rotation=0, ha="right")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                print(sum(gen == 1.) / len(gen) * 100., "% of generated samples at one")
        
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

    elif ORIENT == "landscape":
        fig, axes = plt.subplots(2, 4, figsize=(6, 3.35), sharex='row', sharey='row')

        for i, THRESH in enumerate([0.0, 0.8]):
            bins = np.linspace(THRESH, 1.0, 20)
            hist_kws = dict(density=True, bins=bins, alpha=0.6, edgecolor='k', linewidth=0.4, histtype='stepfilled')

            for j, DOMAIN in enumerate(DOMAINS):
                TRAIN   = os.path.join(env.str("TRAINDIR"), "images", DOMAIN, "storm")
                SAMPLES = os.path.join(env.str("SAMPLES_DIR"), f"{DOMAIN}{VERSION}/gen")
                train = load_pngs(TRAIN)
                gen   = load_pngs(SAMPLES)
                headroom = gen.max() - train.max()
                print(f"Headroom between training and generated samples for {DOMAIN}: {headroom}")
                
                train = train[..., FIELD].ravel()
                train = train[train > THRESH]

                gen = gen[..., FIELD].ravel()
                gen = gen[gen > THRESH]

                ax = axes[i, j]
        
                ax.hist(gen, label="HazGAN", **hist_kws, color="#002147");
                ax.hist(train, label="ERA5", **hist_kws, color="#C8D1DF");
                if i == j == 0:
                    ax.legend(loc="best", frameon=False)
                if i == 0:
                    ax.set_title(f"{DOMAIN.capitalize()}")
                if j == 0:
                    ax.set_ylabel(f"y ≥ {THRESH:.1f}\ndensity", rotation=0, ha="right")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                print(sum(gen == 1.) / len(gen) * 100., "% of generated samples at one")
        
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

# %%
