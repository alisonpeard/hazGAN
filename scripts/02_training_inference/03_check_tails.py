"""
Compare generated and training data tails for different training
configurations.

TODO: clean up
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from environs import Env
from pathlib import Path

from hazGAN import statistics

# script parameters
SCALING   = "rp10000"
DOMAINS   = ["gaussian", "gumbel", "uniform"] #["rescaled", "uniform", "gaussian", "gumbel"] #["uniform", "gaussian", "gumbel", "rescaled"]
VERSION   = ["", "-04", "-05", "-06"][0]
FORMAT    = ["png", "npy"][1]
FIELD     = [0, 1, 2][1]
FIELD_NAME = ['u10', 'tp', 'mslp'][0]
ORIENT    = ["portrait", "landscape"][1]
METHOD    = ["ravel", "max", "pixel"][0]
X0, X1    = 0, 0
UPPER_THRESH = 0.8
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


def load_npys(npy_dir):
    npy_list = os.listdir(npy_dir)
    npy_list = [npy for npy in npy_list if not npy.startswith(".")]
    npy_list = [os.path.join(npy_dir, npy) for npy in npy_list]
    npy_list = sorted(npy_list)

    samples = []
    for npy in (pbar := tqdm(npy_list, desc=f"Loading files from {npy_dir}")):
        pbar.set_description(f"Loading {os.path.basename(npy)}")
        arr = np.load(npy)
        samples.append(arr)
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

    lower = np.quantile(train, THRESH)
    upper = np.max(train)
    ax.plot([lower, upper], [lower, upper], ls="--", color="#666666", lw=0.8, label="1:1 line")

    ax.plot(train_q, gen_q_mean, color="blue", label="data", lw=1.0)
    ax.fill_between(train_q, gen_q_min, gen_q_max, color="blue", alpha=0.3, label="95% CI")

    if i == j == 0:
        ax.legend(loc="best", frameon=False)
    if i == 0:
        ax.set_title(f"{DOMAIN.capitalize()}")
    if j == 0:
        ax.set_ylabel(f"y ≥ q({THRESH:.1f})\ndensity", rotation=0, ha="right")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


if __name__ == "__main__":
    
    env = Env()
    env.read_env()

    fig, axes = plt.subplots(2, 3, figsize=(7, 3.35), sharex='row', sharey='row')

    for j, DOMAIN in enumerate(DOMAINS):
        TRAIN   = Path(env.str("TRAINDIR")) / "images" / SCALING / DOMAIN / FORMAT
        SAMPLES = Path(env.str("SAMPLES_DIR")) / SCALING / f"{DOMAIN}{VERSION}" / FORMAT

        if FORMAT == "npy":
            train   = load_npys(TRAIN)
            gen = load_npys(SAMPLES)
        elif FORMAT == "png":
            train = load_pngs(TRAIN)
            gen = load_pngs(SAMPLES)

        # use image stats to invert scaling
        stats = TRAIN.parent / "image_stats.npz"
        stats = np.load(stats)
        minima = stats["min"]
        maxima = stats["max"]
        method = stats["method"]
        params = stats["param"]
        ranges = maxima - minima
        if method == "minmax":
            print("  Inverting min-max scaling...")
            train = (train / params) * ranges + minima
            gen = (gen / params) * ranges + minima
        elif method == "rp":
            train = train * ranges + minima
            gen = gen * ranges + minima
        print(f"  After inverting {method} scaling: train range [{np.min(train):.2f}, {np.max(train):.2f}]\ngen range [{np.min(gen):.2f}, {np.max(gen):.2f}]")

        if DOMAIN != "gumbel":
            print(f" Train {DOMAIN} range: [{np.min(train):.2f}, {np.max(train):.2f}]")
            print(f"  Gen {DOMAIN} range: [{np.min(gen):.2f}, {np.max(gen):.2f}]")
            train = getattr(statistics, f"inv_{DOMAIN}")(train)
            train = statistics.gumbel(train)
            gen = getattr(statistics, f"inv_{DOMAIN}")(gen)
            print(f"  Train uniform range: [{np.min(train):.2f}, {np.max(train):.2f}]")
            print(f"  Gen uniform range: [{np.min(gen):.2f}, {np.max(gen):.2f}]")
            gen = np.where(gen >= 1., np.nan, gen)
            gen = statistics.gumbel(gen)

        print(f" Train gumbel range: [{np.nanmin(train):.2f}, {np.nanmax(train):.2f}]")
        print(f"  Gen gumbel range: [{np.nanmin(gen):.2f}, {np.nanmax(gen):.2f}]")

        print(f"  After {DOMAIN} transform: train range [{np.nanmin(train):.2f}, {np.nanmax(train):.2f}], gen range [{np.nanmin(gen):.2f}, {np.nanmax(gen):.2f}]")
        
        if METHOD == "ravel":
            train = train[..., FIELD].ravel()
            gen = gen[..., FIELD].ravel()
        elif METHOD == "max":
            train = train[..., FIELD].max(axis=(1, 2))
            gen = gen[..., FIELD].max(axis=(1, 2))
        elif METHOD == "pixel":
            train = train[:, X0, X1, FIELD].squeeze()
            gen = gen[:, X0, X1, FIELD].squeeze()

        for i, THRESH in enumerate([0.0, UPPER_THRESH]):
            lower = np.quantile(train, THRESH)
            upper = np.max(train)
            bins = np.linspace(lower, upper, 20)
            hist_kws = dict(density=True, bins=bins, alpha=0.6, edgecolor='k', linewidth=0.4, histtype='stepfilled')

            ax = axes[i, j]
            histogram(ax, i, j, gen, train, hist_kws)
            # qqplot(ax, i, j, gen[gen>lower], train[train>lower], THRESH)
        
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    
    outpath = os.path.join(env.str("FIG_DIR"), f"tailhists_{FIELD_NAME}_{ORIENT}.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight', transparent=True)


# %%
