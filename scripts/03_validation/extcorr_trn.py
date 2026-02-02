"""
Plot χ(u) for training data.

Notes:
- saves to figures/extcorr/<scaling>/<margins>/train/<fields>/nc/
- Coles, Heffernan, and Tawn (1999, Eq. 3.2)
- plots
    - histograms of ecdfs of subset data (x2)
    - scatter of bivariate Gumbel transformed data
    - χ(u) vs u with 95% bootstrap CI
"""
# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from environs import Env
from pathlib import Path

from hazGAN import statistics
from hazGAN.statistics import ecdf


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 6,
    'font.family': 'monospace'
})


# settings
fields = ["u10", "mslp"] # use string for spatial, list for multivariate
tmin = 0.7
tmax = 0.99
tstep = 0.01
nboot = 200
t_final = 0.8


def subset(data, field:str, thresh:float=15):
    maxima = data.sel(field=field)["anomaly"].max(dim=["lat", "lon"])
    mask = (maxima > thresh).values
    print(f"Subsetting to {mask.sum().item()} samples above threshold {thresh}")
    return data.isel(time=mask)


def sample_pixels(data, field:str, var:str="anomaly", labels:dict={}):
    h, w = data.sizes["lat"], data.sizes["lon"]
    i, j = np.random.randint(0, h), np.random.randint(0, w)
    arr0 = data.sel(field=field).isel(lat=i, lon=j)[var].values
    labels[0] = f"({i}, {j})"

    i, j = np.random.randint(0, h), np.random.randint(0, w)
    arr1 = data.sel(field=field).isel(lat=i, lon=j)[var].values
    labels[1] = f"({i}, {j})"
    return arr0, arr1, labels


def sample_fields(data, fields:list[str], var:str="anomaly", labels:dict={}):
    h, w = data.sizes["lat"], data.sizes["lon"]
    i, j = np.random.randint(0, h), np.random.randint(0, w)

    arr0 = data.sel(field=fields[0]).isel(lat=i, lon=j)[var].values

    arr1 = data.sel(field=fields[1]).isel(lat=i, lon=j)[var].values
    labels[0] = f"({i},{j}): {fields[0]}"
    labels[1] = f"{fields[1]}"
    return arr0, arr1, labels


def sample(data, fields:str, var:str="anomaly"):
    if isinstance(fields, list):
        return sample_fields(data, fields, var=var)
    else:
        return sample_pixels(data, fields, var=var)


def chi(u, v, t=0.8):
    """Coles (2001) §8.4, u,v~Unif[0,1]"""
    n = len(u)
    both_below = np.sum((u < t) & (v < t))
    prob_below = both_below / n
    if prob_below > 0:
        chi = 2 - np.log(prob_below) / np.log(t)
    else:
        chi = np.nan
    return chi


# load the data
if __name__ == "__main__":
    env = Env()
    env.read_env()

    datadir = Path(env.str("TRAINDIR"))
    figdir = Path(env.str("FIG_DIR")) / "extcorr" / "train"

    print(f"\nLoading data from {datadir / 'data.nc'}")
    print(f"Saving figures to {figdir}\n")

    data = xr.open_dataset(datadir / "data.nc")
    data = subset(data, "u10", thresh=15) # always subset by wind

    thresholds = np.arange(tmin, tmax, tstep)

    figdir = figdir / "_".join(fields) if isinstance(fields, list) else figdir / fields
    figdir.mkdir(parents=True, exist_ok=True)
    print("Made figdir:", figdir)

    for i in range(30):
        np.random.seed(42 + i)
        x0, x1, labels = sample(data, fields)
        u0, u1 = ecdf(x0)(x0), ecdf(x1)(x1)

        chis = np.zeros((nboot, len(thresholds)), dtype=float)

        for b in range(nboot):

            idx = np.random.choice(len(u0), size=len(u0), replace=True)

            u0_b, u1_b = u0[idx], u1[idx]
            
            for j, u in enumerate(thresholds):
                chis[b, j] = chi(u0_b, u1_b, t=u)

        chis_mean = chis.mean(axis=0)
        ci_lower = np.percentile(chis, 2.5, axis=0)
        ci_upper = np.percentile(chis, 97.5, axis=0)
        chi_final = chi(u0, u1, t=t_final)

        y0, y1 = statistics.gumbel(u0), statistics.gumbel(u1)

        fig, axs = plt.subplots(2, 2, figsize=(3, 2), constrained_layout=True)
        ax = axs[0, 0]
        ax.hist(u0, bins=30, density=True, alpha=0.7, edgecolor='k', color='white', linewidth=0.5)

        ax = axs[0, 1]
        ax.hist(u1, bins=30, density=True, alpha=0.7, edgecolor='k', color='white', linewidth=0.5)

        ax = axs[1, 0]
        ax.scatter(y0, y1, s=5, facecolor='k', edgecolor='none')
        ax.set_xlabel("y0 ~ Gumbel")
        ax.set_ylabel("y1 ~ Gumbel")

        ax = axs[1, 1]
        ax.plot(thresholds, chis_mean, '-', color='k', lw=0.5)
        ax.fill_between(thresholds, ci_lower, ci_upper,
                        color='gray', alpha=0.25, linewidth=0.1)

        
        ax.spines["bottom"].set_position("zero")
        ax.set_xlim(0.7, 1.0)

        ax.vlines(t_final, 0, chi_final, color='r', lw=0.5, ls='--')
        ax.hlines(chi_final, 0.6, t_final, color='r', lw=0.5, ls='--')
        ax.text(0.8, 1.0, f"χ={chi_final:.2f}", color='r', fontsize=6)

        ax.set_xlabel("u")
        ax.set_ylabel("χ(u)")

        fig.suptitle(f"{labels[0]} & {labels[1]}")
        fig.savefig(figdir / f"{str(i).zfill(2)}.png", dpi=300)

        if i > 3:
            plt.close(fig)

    print(f"Saved {i+1} diagnostic plots to {figdir}")

# %%
