"""
QQ plot comparison of generated vs training data across marginal transforms.
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

# parameters
SCALING = "rp10000"
DOMAINS = ["gaussian", "gumbel", "uniform"]
FIELD = 1
FIELD_NAME = "u10"
THRESH = 0.8
N_PIXELS = 500
N_BOOT = 100
N_QUANTILES = 100

plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 6
})


def load_pngs(png_dir):
    png_list = sorted([f for f in os.listdir(png_dir) if not f.startswith(".")])
    samples = []
    for png in tqdm(png_list, desc=f"Loading {png_dir.name}"):
        with Image.open(png_dir / png) as img:
            samples.append(np.array(img))
    samples = np.array(samples).astype(float) / 255.
    return samples


def invert_scaling(data, stats_path):
    stats = np.load(stats_path)
    minima, maxima = stats["min"], stats["max"]
    ranges = maxima - minima
    method, params = stats["method"], stats["param"]
    if method == "minmax":
        return (data / params) * ranges + minima
    elif method == "rp":
        return data * ranges + minima
    return data


def to_gumbel(data, domain):
    if domain == "gumbel":
        return data
    data = getattr(statistics, f"inv_{domain}")(data)
    data = np.where(data >= 1., np.nan, data)
    return statistics.gumbel(data)


def quantiles_boot(x, nboot, bootsize, nquantiles):
    x = x[~np.isnan(x)]
    quantiles = np.linspace(0, 1, nquantiles)
    qdist = np.array([
        np.quantile(np.random.choice(x, size=bootsize, replace=True), quantiles)
        for _ in range(nboot)
    ])
    return qdist.mean(axis=0), np.percentile(qdist, 2.5, axis=0), np.percentile(qdist, 97.5, axis=0)


def qqplot(ax, gen, train, show_legend=False, title=None, ylabel=None):
    quantiles = np.linspace(0, 1, N_QUANTILES)
    train_clean = train[~np.isnan(train)]
    gen_clean = gen[~np.isnan(gen)]
    
    if len(train_clean) == 0 or len(gen_clean) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    train_q = np.quantile(train_clean, quantiles)
    bootsize = min(len(gen_clean), 1000)
    gen_mean, gen_lo, gen_hi = quantiles_boot(gen_clean, N_BOOT, bootsize, N_QUANTILES)

    lo, hi = train_q.min(), train_q.max()
    ax.plot([lo, hi], [lo, hi], ls="--", color="#666666", lw=0.8, label="1:1")
    ax.plot(train_q, gen_mean, color="blue", lw=1.0, label="data")
    ax.fill_between(train_q, gen_lo, gen_hi, color="blue", alpha=0.3, label="95% CI")

    if show_legend:
        ax.legend(loc="best", frameon=False)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel, rotation=0, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def extract_pixels(data, pixels, field):
    """Extract values at specified pixel locations."""
    return np.array([data[:, px, py, field] for px, py in pixels]).T.ravel()


def extract_max(data, field):
    """Extract spatial maximum per sample."""
    return np.nanmax(data[..., field], axis=(1, 2))


if __name__ == "__main__":
    env = Env()
    env.read_env()

    np.random.seed(42)
    pixels = np.random.randint(0, 64, size=(N_PIXELS, 2))

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(7, 2.5), sharex='row', sharey='row')

    for j, domain in enumerate(DOMAINS):
        train_dir = Path(env.str("TRAINDIR")) / "images" / SCALING / domain / "png"
        gen_dir = Path(env.str("SAMPLES_DIR")) / SCALING / domain / "png"
        stats_path = train_dir.parent / "image_stats.npz"

        train = load_pngs(train_dir)
        gen = load_pngs(gen_dir)

        train = invert_scaling(train, stats_path)
        gen = invert_scaling(gen, stats_path)

        train = to_gumbel(train, domain)
        gen = to_gumbel(gen, domain)

        # row 0: pixel average
        train_px = extract_pixels(train, pixels, FIELD)
        gen_px = extract_pixels(gen, pixels, FIELD)
        thresh_val = np.nanquantile(train_px, THRESH)
        qqplot(
            axes[j],
            gen_px[gen_px >= thresh_val],
            train_px[train_px >= thresh_val],
            show_legend=(j == 0),
            title=domain.capitalize(),
            ylabel=f"pixel\ny ≥ q({THRESH})" if j == 0 else None
        )

    fig.tight_layout()
    outpath = Path(env.str("FIG_DIR")) / f"qqplot_{FIELD_NAME}.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved to {outpath}")
# %%