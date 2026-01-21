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
SCALING = "rp10000"
DOMAINS   = ["gaussian"] #["rescaled", "uniform", "gaussian", "gumbel"] #["uniform", "gaussian", "gumbel", "rescaled"]
VERSION   = ["", "-04", "-05", "-06"][0]
FIELD     = [0, 1, 2][1]
FIELD_NAME = ['u10', 'tp', 'mslp'][0]
ORIENT    = ["portrait", "landscape"][1]
METHOD    = ["ravel", "max", "pixel"][-1]
plt.rcParams.update({'font.family': 'monospace', 'font.size': 6, 'axes.labelsize': 7, 'axes.titlesize': 7, 'legend.fontsize': 6})


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
    return samples


def histogram(ax, i, j, gen, train, hist_kws):
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
    print(f"  {sum(gen == 1.) / len(gen) * 100.:.2f}% of generated samples at one")


def quantiles_boot(x, nboot, bootsize, nquantiles=1000):
    quantiles = np.linspace(0, 1, nquantiles)
    qdist = []
    for _ in range(nboot):
        xboot = np.random.choice(x, size=bootsize, replace=True)
        qdist.append(np.quantile(xboot, quantiles))
    qdist = np.array(qdist)
    return np.mean(qdist, axis=0), np.percentile(qdist, 2.5, axis=0), np.percentile(qdist, 97.5, axis=0)


def qqplot(ax, i, j, gen, train, THRESH):
    quantiles = np.linspace(0, 1, 1000)
    train_q = np.quantile(train, quantiles)
    print(f"Making 100 bootstrap samples of size {len(train)} for quantile CI estimation...")
    gen_q_mean, gen_q_min, gen_q_max = quantiles_boot(gen, nboot=100, bootsize=len(train), nquantiles=1000)

    ax.plot([THRESH, 1], [THRESH, 1], ls="--", color="#666666", lw=0.8, label="1:1 line")

    ax.plot(train_q, gen_q_mean, color="blue", label="data", lw=1.0)
    ax.fill_between(train_q, gen_q_min, gen_q_max, color="blue", alpha=0.3, label="95% CI")

    if i == j == 0:
        ax.legend(loc="lower right", frameon=False)
    if i == 0:
        ax.set_title(f"{DOMAIN.capitalize()}")
    if j == 0:
        ax.set_ylabel(f"y ≥ {THRESH:.1f}\ndensity", rotation=0, ha="right")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

if __name__ == "__main__":
    
    env = Env()
    env.read_env()

    if ORIENT == "portrait":

        fig, axes = plt.subplots(4, 2, figsize=(3.35, 6), sharex='col', sharey='col')

        for j, THRESH in enumerate([0.0, 0.8]):
            bins = np.linspace(THRESH, 1.0, 20)
            hist_kws = dict(density=True, bins=bins, alpha=0.6, edgecolor='k', linewidth=0.4, histtype='stepfilled')

            for i, DOMAIN in enumerate(DOMAINS):
                TRAIN   = os.path.join(env.str("TRAINDIR"), "images", SCALING, DOMAIN)
                SAMPLES = os.path.join(env.str("SAMPLES_DIR"), SCALING, f"{DOMAIN}{VERSION}", "png")
                train = load_pngs(TRAIN)
                gen   = load_pngs(SAMPLES)
                print(f"  Loaded {gen.shape} samples for {DOMAIN}.")
                headroom = gen.max() - train.max()
                print(f"  Headroom between training and generated samples for {DOMAIN}: {headroom:.4f}")
                
                train = train[..., FIELD].ravel()
                gen = gen[..., FIELD].ravel()

                ax = axes[i, j]
                histogram(ax, i, j, gen, train, hist_kws)
        
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

    elif ORIENT == "landscape":

        fig, axes = plt.subplots(2, 4, figsize=(7, 3.35), sharex='row', sharey='row')

        for j, DOMAIN in enumerate(DOMAINS):
            TRAIN   = os.path.join(env.str("TRAINDIR"), "images", SCALING, DOMAIN, "png")
            SAMPLES = os.path.join(env.str("SAMPLES_DIR"), SCALING, f"{DOMAIN}{VERSION}", "png")
            train = load_pngs(TRAIN)
            gen   = load_pngs(SAMPLES)
            print(f"  Loaded {gen.shape} samples for {DOMAIN}.")
            headroom = gen.max() - train.max()
            print(f"  Headroom between training and generated samples for {DOMAIN}: {headroom:.4f}")
            
            if METHOD == "ravel":
                # all pixels histogram
                train = train[..., FIELD].ravel()
                gen = gen[..., FIELD].ravel()
            elif METHOD == "max":
                # maxima-only histogram
                train = train[..., FIELD].max(axis=(1, 2))
                gen = gen[..., FIELD].max(axis=(1, 2))
            elif METHOD == "pixel":
                # single pixel histogram
                train = train[:, 32, 32, FIELD].squeeze()
                gen = gen[:, 32, 32, FIELD].squeeze()

            for i, THRESH in enumerate([0.0, 0.8]): # [0.0. [0.8]]
                bins = np.linspace(THRESH, 1.0, 20)
                hist_kws = dict(density=True, bins=bins, alpha=0.6, edgecolor='k', linewidth=0.4, histtype='stepfilled')

                ax = axes[i, j]
                # histogram(ax, i, j, gen, train, hist_kws)
                qqplot(ax, i, j, gen[gen>THRESH], train[train>THRESH], THRESH)
            
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    outpath = os.path.join(env.str("FIG_DIR"), f"tailhists_{FIELD_NAME}_{ORIENT}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)


# %%
