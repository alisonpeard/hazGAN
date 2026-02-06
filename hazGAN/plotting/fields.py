import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from .base import makegrid
from .base import contourmap
from .base import CMAP
from ..statistics import get_extremal_coeffs_nd


def pearson(array):
    def pixelcorr(array, i):
        array = array[:, i, :].copy()
        corr  = np.corrcoef(array.T)
        return corr
    
    _, h, w, c = array.shape
    array = array.reshape(-1, h * w, c)

    corrs = []
    for i in range(h * w):
        corrs.append(pixelcorr(array, i))
    corrs = np.stack(corrs, axis=0).reshape(h, w, c, c)
    return corrs[..., 0, 1]


def smith1990(array):
    def get_ext_coefs(x):
        _, h, w, _ = x.shape
        excoefs = get_extremal_coeffs_nd(x, [*range(h * w)])
        excoefs = np.array([*excoefs.values()]).reshape(h, w)
        return excoefs
    
    return get_ext_coefs(array)


@njit
def _chi(u, v, t=0.9):
    """https://doi.org/10.1023/A:1009963131610"""
    n = len(u)
    both_above = np.sum((u > t) & (v > t))
    prob_above = both_above / n
    pu_above = np.sum(u > t) / n
    if both_above < 3:
        return np.nan
    return prob_above / pu_above


def extcorr(array):
    _, h, w, c = array.shape
    array = array.reshape(-1, h * w, c)
    extcorrs = []
    for i in range(h * w):
        u, v = array[:, i, 0], array[:, i, 1]
        chi = _chi(u, v)
        extcorrs.append(chi)
    extcorrs = np.stack(extcorrs, axis=0).reshape(h, w)
    return extcorrs


@njit(parallel=True)
def extcorrboot(array, nboot:int=100, size:int=150):
    n, h, w, c = array.shape
    hw = h * w
    # array = array.reshape(n, hw, c)
    array = np.reshape(array.copy(), (n, hw, c))
    extcorrs = np.zeros(hw)
    for i in prange(hw):
        u, v = array[:, i, 0], array[:, i, 1]
        chi = 0.
        for _ in range(nboot):
            idx = np.random.choice(n, size=size, replace=True)
            u_samp = u[idx]
            v_samp = v[idx]
            chival = _chi(u_samp, v_samp)
            if not np.isnan(chival):
                chi += chival
        extcorrs[i] = chi / nboot
    return np.reshape(extcorrs, (h, w))


def plot(fake, train, func, fields=[0, 1], figsize=1.,
         cmap=CMAP, vmin=None, vmax=None,
         title="", cbar_label="", **func_kws
         ) -> plt.Figure:
    """
    Plot relationships between climate fields.

    Args:
        fake: model data of size _ x h x w c
        train: trianing data of size _ x h x w x c
        func: function to use to measure dependence
        fields: which fields to compare (pairs only)
    """
    train = train[..., fields]
    fake  = fake[..., fields]

    train_res = func(train, **func_kws)
    fake_res  = func(fake, **func_kws)

    vmin = vmin or np.nanmin(train_res)
    vmax = vmax or np.nanmax(train_res)
    cmap = getattr(plt.cm, cmap)

    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, axs, cax = makegrid(1, 2, figsize=figsize, cbar_width=0.1)
    im = contourmap(train_res, ax=axs[0], vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.2)
    _  = contourmap(fake_res, ax=axs[-1], vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.2)

    axs[0].set_title("ERA5", y=-0.2)
    axs[-1].set_title("HazGAN", y=-0.2)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, rotation=0, labelpad=10)
    fig.suptitle(title, y=1.05)

    # TODO: look for alternatives!
    corr = np.corrcoef(train_res.flatten(), fake_res.flatten())[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    mae = np.mean(np.abs(train_res - fake_res))
    print(f"Mean Absolute Error: {mae:.4f}")

    return fig, {"mae": mae, "pearson": corr}