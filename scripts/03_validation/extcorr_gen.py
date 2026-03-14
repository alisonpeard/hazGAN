"""
Plot χ(u) (extremal correlation) for generated data.

Notes:
- saves to figures/extcorr/<scaling>/<margins>/gen/<fields>/nc/
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
from numba import njit

from hazGAN import statistics


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.titleweight': 'normal',
    'legend.fontsize': 6,
    'font.family': 'sans-serif'
})


# settings
fields = ["u10", "tp"] # use string for spatial, list for multivariate
method = "chi" # try 'chibar' to investigate asymptotic independence
margins = "gumbel" # "rescaled", "uniform", "gaussian", "gumbel"
scaling = "rp10000"
tmin = 0.7
tmax = 0.99
tstep = 0.01
nboot = 200
t_final = 0.9


def subset(data, field:str, thresh:float=15):
    maxima = data.sel(field=field)["anomaly"].max(dim=["lat", "lon"])
    mask = (maxima > thresh).values
    print(f"Subsetting to {mask.sum().item()} samples above threshold {thresh}")
    return data.isel(time=mask)


def sample_pixels(data, field:str, var:str="anomaly", labels:dict={}):
    h, w = data.sizes["lat"], data.sizes["lon"]
    i, j = np.random.randint(0, h), np.random.randint(0, w)
    arr0 = data.sel(field=field).isel(lat=i, lon=j)[var].values
    labels[0] = f"({i},{j})"

    i, j = np.random.randint(0, h), np.random.randint(0, w)
    arr1 = data.sel(field=field).isel(lat=i, lon=j)[var].values
    labels[1] = f"({i},{j})"
    return arr0, arr1, labels


def sample_fields(data, fields:list[str], var:str="anomaly", labels:dict={}):
    h, w = data.sizes["lat"], data.sizes["lon"]
    i, j = np.random.randint(0, h), np.random.randint(0, w)

    arr0 = data.sel(field=fields[0]).isel(lat=i, lon=j)[var].values

    arr1 = data.sel(field=fields[1]).isel(lat=i, lon=j)[var].values
    labels[0] = f"({i}, {j}): {fields[0]}"
    labels[1] = f"{fields[1]}"
    return arr0, arr1, labels


def sample(data, fields:str, var:str="anomaly"):
    if isinstance(fields, list):
        return sample_fields(data, fields, var=var)
    else:
        return sample_pixels(data, fields, var=var)


@njit
def _ecdf(x: np.ndarray) -> np.ndarray:
    """R ecdf implementation with Weibull plotting positions."""
    n = len(x)
    sorted_x = np.sort(x)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        rank = np.searchsorted(sorted_x, x[i], side='right')
        result[i] = rank / (n + 1)  
    return result


def chi(u, v, t):
    """https://doi.org/10.1023/A:1009963131610"""
    n = len(u)
    both_above = np.sum((u > t) & (v > t))
    prob_above = both_above / n
    pu_above = np.sum(u > t) / n
    if both_above < 3:
        return np.nan
    return prob_above / pu_above


def chibar(u, v, t):
    """https://doi.org/10.1023/A:1009963131610"""
    n = len(u)
    both_above = np.sum((u > t) & (v > t))
    u_above = np.sum(u > t)
    if both_above < 3:
        return np.nan
    pboth_above = both_above / n
    pu_above = u_above / n
    return 2 * np.log(pu_above) / np.log(pboth_above) - 1


def extcorr(u, v, t=0.9, method="chi"):
    u = _ecdf(u)
    v = _ecdf(v)
    if method == "chi":
        return chi(u, v, t=t)
    elif method == "chibar":
        return chibar(u, v, t=t)
    else:
        raise ValueError(f"Invalid method: {method}")


def bootstrap_stats(data, confidence=0.99):
    from scipy import stats
    mask = ~np.isnan(data)
    n = mask.sum(axis=0)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, ddof=1, axis=0)
    se = std / np.sqrt(n)
    tcrit = stats.t.ppf(0.5 + confidence / 2, df=n-1)
    moe = tcrit * se
    return mean, mean - moe, mean + moe, n


# load the data
if __name__ == "__main__":
    env = Env()
    env.read_env()

    datadir = Path(env.str("GENDIR")) / scaling / margins / "nc"
    figdir = Path(env.str("FIGDIR")) / method / "gen" / scaling / margins

    print(f"\nLoading data from {datadir / 'data.nc'}")
    print(f"Saving figures to {figdir}\n")

    data = xr.open_dataset(datadir / "data.nc")

    thresholds = np.arange(tmin, tmax, tstep)

    figdir = figdir / "_".join(fields) if isinstance(fields, list) else figdir / fields
    figdir.mkdir(parents=True, exist_ok=True)

    for i in range(30):
        np.random.seed(42 + i)
        u0, u1, labels = sample(data, fields, var="uniform")
        chis = np.zeros((nboot, len(thresholds)), dtype=float)

        for b in range(nboot):
            idx = np.random.choice(len(u0), size=150, replace=True)
            u0_b, u1_b = u0[idx], u1[idx]
            
            for j, u in enumerate(thresholds):
                chis[b, j] = extcorr(u0_b, u1_b, t=u, method=method)

        # bootstrap stats
        chis_mean, ci_lower, ci_upper, n = bootstrap_stats(chis)

        # get χ at t_final
        t_final_idx = np.argmin(np.abs(thresholds - t_final))
        chi_final = chis_mean[t_final_idx]

        y0, y1 = statistics.gumbel(u0), statistics.gumbel(u1)

        y_final = statistics.gumbel(t_final)
        ymax = max(y0.max(), y1.max()) * 1.05
        ymin = min(y0.min(), y1.min()) * 0.95

        fig, axs = plt.subplots(1, 2, figsize=(3, 1.5), constrained_layout=True)

        ax = axs[0]
        ax.scatter(y0, y1, s=5, facecolor='k', edgecolor='none')
        ax.set_xlabel("y0 ~ Gumbel")
        ax.set_ylabel("y1 ~ Gumbel")
        ax.hlines(y_final, xmin=ymin, xmax=ymax, color='r', lw=0.5, ls='--', zorder=0)
        ax.vlines(y_final, ymin=ymin, ymax=ymax, color='r', lw=0.5, ls='--', zorder=0)
        ax.fill_between([y_final, ymax], y_final, ymax, color='gray', alpha=0.3,
                zorder=0, edgecolor='none')

        ax = axs[1]
        # valid = n > 10
        # ax.plot(thresholds[valid], chis_mean[valid], '-', color='k', lw=0.5)
        # ax.plot(thresholds[~valid], chis_mean[~valid], '--', color='k', lw=0.5)
        ax.plot(thresholds, chis_mean, '-', color='k', lw=0.5)
        ax.fill_between(thresholds, ci_lower, ci_upper,
                        color='gray', alpha=0.25, linewidth=0.1)
        

        if method == "chi":
            ax.set_ylim(-0.05, 1.05)
        elif method == "chibar":
            ax.set_ylim(-1.05, 1.05)
            ax.spines["bottom"].set_position("zero")
        ax.set_xlim(0.7, 1.0)

        if not np.isnan(chi_final):
            ax.vlines(t_final, 0, chi_final, color='r', lw=0.5, ls='--')
            ax.hlines(chi_final, 0.6, t_final, color='r', lw=0.5, ls='--')
            ax.text(0.8, np.nanmax(chis_mean), f"χ={chi_final:.2f}",
                    color='r', fontsize=6, fontweight='bold')

        ax.set_xlabel("u")
        ax.set_ylabel("χ(u)")

        fig.suptitle(f"{labels[0]} & {labels[1]}")
        fig.savefig(figdir / f"{str(i).zfill(2)}.png", dpi=300)

        if i > 3:
            plt.close(fig)

    print(f"Saved {i+1} diagnostic plots to {figdir}")

# %%

