# %%
import numpy as np
import matplotlib.pyplot as plt

from .base import makegrid
from .base import contourmap
from .base import heatmap
from .base import CMAP
from ..statistics import get_extremal_coeffs_nd


def plot(fake, train, func, fields=[0, 1], figsize=1.,
         cmap=CMAP, vmin=None, vmax=None,
         title="Untitled", cbar_label="", **func_kws
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

    fig, axs, cax = makegrid(1, 2, figsize=figsize)
    im = contourmap(train_res, ax=axs[0], vmin=vmin, vmax=vmax, cmap=cmap)
    _  = contourmap(fake_res, ax=axs[-1], vmin=vmin, vmax=vmax, cmap=cmap)

    axs[0].set_title("ERA5", y=-0.15)
    axs[-1].set_title("HazGAN", y=-0.15)

    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.suptitle(title, y=1.05)

    corr = np.corrcoef(train_res.flatten(), fake_res.flatten())[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    mae = np.mean(np.abs(train_res - fake_res))
    print(f"Mean Absolute Error: {mae:.4f}")

    return fig, {"mae": mae, "pearson": corr}


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


def tail_dependence(array):
    _, h, w, c = array.shape
    array = array.reshape(-1, h * w, c)

    coeffs = []
    for i in range(h * w):
        x, y = array[:, i, 0], array[:, i, 1]
        chi = _tail_dependence_coeff(x, y)
        coeffs.append(chi)
    coeffs = np.stack(coeffs, axis=0).reshape(h, w)
    return coeffs


def _tail_dependence_coeff(u, v):
    """
    Classical tail dependence coefficient λ for upper tail dependence.

    Args:
        u, v: 1D arrays of uniform marginals
        tail: 'upper' or 'lower'
    Returns:
        λ: tail dependence coefficient

    Refs:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman & Hall.
        Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
    """
    thresholds = np.arange(0.8, 0.99, 0.01)  # Multiple thresholds
    lambdas = []
    
    for t in thresholds:
        u_exceed = u > t
        both_exceed = (u > t) & (v > t)
        
        if np.sum(u_exceed) > 0:
            lambda_t = np.sum(both_exceed) / np.sum(u_exceed)
            lambdas.append(lambda_t)

    return np.mean(lambdas) if lambdas else 0

# %%
