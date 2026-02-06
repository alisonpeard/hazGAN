import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

from .base import makegrid
from .base import contourmap


def pearson(array):
    n, h, w = array.shape
    array = array.reshape(n, h * w)
    corrs = np.corrcoef(array.T)
    print(corrs.shape)
    return corrs


@njit
def _chi2(u, v, t=0.9):
    """Coles (2001) §8.4, u,v~Unif[0,1]
    NOTE: logs can make this unstable
    """
    n = len(u)
    both_below = np.sum((u < t) & (v < t))
    prob_below = both_below / n
    u_below = np.sum(u < t) / n
    if prob_below > 0:
        chi = 2 - np.log(prob_below) / np.log(u_below)
    else:
        chi = np.nan
    return chi


@njit
def _chi1(u, v, t=0.9):
    n = len(u)
    both_above = np.sum((u > t) & (v > t))
    prob_above = both_above / n
    prob_u = np.sum(u > t) / n
    if prob_u > 0:
        chi = prob_above / prob_u
    else:
        chi = np.nan
    return chi



@njit(parallel=True)
def extcorr(array):
    _, h, w = array.shape
    array = array.reshape(-1, h * w)

    extcorrs = np.empty((h * w, h * w))
    for i in prange(h * w):
        for j in range(i):
            u, v = array[:, i], array[:, j]
            chi = _chi1(u, v)
            extcorrs[i, j] = chi
            extcorrs[j, i] = chi

    return extcorrs


@njit(parallel=True)
def extcorrboot(array, nboot=100, size=150):
    n, h, w = array.shape
    array = array.reshape(-1, h * w)

    extcorrs = np.zeros((h * w, h * w))
    
    np.random.seed(42)
    for i in prange(h * w):
        for j in range(i):
            u, v = array[:, i], array[:, j]
            for _ in range(nboot):
                idx = np.random.choice(n, size=size, replace=True)
                u_samp = u[idx]
                v_samp = v[idx]
                chi = _chi1(u_samp, v_samp)

                if np.isnan(chi):
                    continue
                    
                extcorrs[i, j] += chi
                extcorrs[j, i] += chi

    return extcorrs / nboot


def plot(fake, train, func, field=0, figsize=1.,
         cmap="viridis", vmin=None, vmax=None,
         title="", cbar_label="",
         alpha:float=1e-4, **func_kws):
    train = train[..., field].copy()
    fake  = fake[..., field].copy()
    
    train_res = func(train)
    fake_res  = func(fake)

    if alpha is not None:
        vmin = vmin or np.nanquantile(np.concatenate([fake_res, train_res]), alpha)
        vmax = vmax or np.nanquantile(np.concatenate([fake_res, train_res]), 1-alpha)
    else:
        vmin = vmin or train_res.min()
        vmax = vmax or train_res.max()
    
    cmap = getattr(plt.cm, cmap)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, axs, cax = makegrid(1, 2, figsize=figsize, cbar_width=0.1)

    im = contourmap(
        train_res, ax=axs[0], vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.2, features=False
    )
    _  = contourmap(
        fake_res, ax=axs[-1], vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.2, features=False
    )

    for ax in axs:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    axs[0].set_title("ERA5", y=-0.2)
    axs[-1].set_title("HazGAN", y=-0.2)
    fig.suptitle(title, y=1.05)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, rotation=0, labelpad=10)
    fig.suptitle(title, y=1.05)

    corr = np.corrcoef(train_res.flatten(), fake_res.flatten())[0, 1]
    mae = np.mean(np.abs(train_res - fake_res))

    return fig, {"mae": mae, "pearson": corr}