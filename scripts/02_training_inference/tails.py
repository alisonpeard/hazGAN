"""
Notes:
- optionally transform all to gumbel margins to compare
- option to view single pixel or ravelled data or maxima
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environs import Env
from pathlib import Path

from hazGAN import statistics


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 6,
    'font.family': 'sans-serif'
})


# script parameters
scaling   = "rp10000"
domains   = ["uniform", "gumbel", "gaussian"]
field_idx = 0
field_nom = "u10"
tmax = 0.95
hist_bins = 20

trn_kws = dict(
    density=True,
    alpha=0.8,
    color="#A6CEE3",
    histtype="stepfilled",
    joinstyle="round"

)

gen_kws = dict(
    density=True,
    color="#D55E00",
    histtype="step",
    alpha=1.0,
    lw=1.3,
    joinstyle="round"
)


def load_npys(npy_dir):
    npy_list = os.listdir(npy_dir)
    npy_list = [npy for npy in npy_list if not npy.startswith(".")]
    npy_list = [os.path.join(npy_dir, npy) for npy in npy_list]
    npy_list = sorted(npy_list)

    samples = []
    for npy in (pbar := tqdm(npy_list, desc=f"Loading files from {npy_dir}", leave=False)):
        pbar.set_description(f"Loading {os.path.basename(npy)}")
        arr = np.load(npy)
        samples.append(arr)
    samples = np.array(samples).astype(float)
    samples /= 255.
    return samples


def undo_scaling(x:np.array, stats:dict) -> np.array:
    mins = stats["min"]
    maxs = stats["max"]
    method = stats["method"]
    params = stats["param"]
    ranges = maxs - mins

    if method == "minmax":
        return (x / params) * ranges + mins
    elif method == "rp":
        return x * ranges + mins


def transform_to_gumbel(x:np.array, domain:str="gumbel") -> np.array:
    u = getattr(statistics, "inv_" + domain)(x)
    if (u >= 1.).any():
        print(f"WARNING: u ≥ 1 encountered for {domain}.")
    u = np.where(u >= 1., np.nan, u)
    y = statistics.gumbel(u)
    return y


def histogram(ax, i, j, thresh, train, gen, trn_kws, gen_kws):
    ax.hist(train, label=f"ERA5", **trn_kws);
    ax.hist(gen, label=f"HazGAN", **gen_kws);

    if (i == 0) & (j == 2):
        ax.legend(loc="upper right", frameon=False)
    if i == 0:
        ax.set_title(f"{domain.capitalize()}")
    if (j == 0) & (i == 0):
        ax.set_ylabel(f"Density\n\ny ≥ q{100 * thresh:.0f}", rotation=0, ha="right")
    if (j == 0) & (i == 1):
        ax.set_ylabel(f"\n\ny ≥ q{100 * thresh:.0f}", rotation=0, ha="right")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save_stats_text(path, stats_dict):
    with open(path, "w") as f:
        f.write(f"RESULTS SUMMARY\n{'='*20}\n\n")
        for section, values in stats_dict.items():
            f.write(f"[{section}]\n")
            for k, v in values.items():
                val_str = f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)
                f.write(f"{k:.<20} {val_str}\n")
            f.write("\n")


if __name__ == "__main__":
    
    env = Env()
    env.read_env()

    # one figure for everything
    fig, axes = plt.subplots(2, 3, figsize=(6, 2.5), constrained_layout=True)

    trndir = Path(env.str("TRAINDIR")) / "images" / scaling
    gendir = Path(env.str("SAMPLES_DIR")) / scaling
    figdir = Path(env.str("FIG_DIR")) / "tails" / scaling
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{trndir=}")
    print(f"{gendir=}\n")

    results = {}

    # loop through domains
    for j, domain in enumerate(domains):

        results[domain] = {}

        # configure the paths
        trndir_j =  trndir / domain / "npy"
        gendir_j = gendir / domain / "npy"
        stats_j = trndir_j.parent / "image_stats.npz"

        print(f"Loading {Path(domain) / 'npy'}")

        # load all the data
        train = load_npys(trndir_j)
        gen = load_npys(gendir_j)
        stats = np.load(stats_j)

        # undo the (0, 1) scaling
        train = undo_scaling(train, stats)
        gen = undo_scaling(gen, stats)

        results[domain]["train_max"] = np.nanmax(train)
        results[domain]["gen_max"] = np.nanmax(gen)

        # convert everything to gumbel margins
        train = transform_to_gumbel(train, domain)
        gen = transform_to_gumbel(gen, domain)

        results[domain]["train_gumbel_max"] = np.nanmax(train)
        results[domain]["gen_gumbel_max"] = np.nanmax(gen)
        
        # flatten the arrays
        train = train[..., field_idx].ravel()
        gen = gen[..., field_idx].ravel()

        # plot histograms at two thresholds (rows)
        for i, t in enumerate([0.0, tmax]):

            # construct the histogram bins
            lower = np.nanquantile(train, t)
            upper = max(np.nanmax(train), np.nanmax(gen))
            bins = np.linspace(lower, upper, hist_bins)
            trn_kws.update(dict(bins=bins))
            gen_kws.update(dict(bins=bins))

            histogram(axes[i, j], i, j, t, train, gen, trn_kws, gen_kws)
        
    
    # save the figure
    figpath = figdir / f"{field_nom}.pdf"
    plt.savefig(figpath, dpi=300, transparent=True)
    print(f"Saved figure to {figpath}\n")

    # save the results summary
    respath = figdir / f"{field_nom}.txt"
    save_stats_text(respath, results)
    print(f"Saved results summary to {respath}\n")

# %%
