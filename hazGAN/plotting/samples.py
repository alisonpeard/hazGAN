# %%
import numpy as np
from .base import CMAP
from .base import makegrid, contourmap
from ..statistics import invPIT


def identity(array, *args, **kwargs):
    return array


def gumbel(array, axis=0):
    array = np.clip(array, 1e-6, 1 - 1e-6)
    return -np.log(-np.log(array))


def anomaly(array, reference, params):
    array = invPIT(array, reference, params)
    return array


def plot(fake, train, field=0, transform=None, vmin=None, vmax=None, cmap=CMAP, title="Untitled",
        cbar_label='', cbar_width=0.25, linewidth=.15, alpha=1e-4, alpha_vlim=True, 
        nrows=4, ncols=8, cbar_formatter=None, **transform_kws):
    """Plot training samples on top row and generated samples on bottom row."""

    transform = transform or identity
    fake  = transform(fake, **transform_kws)
    train = transform(train, **transform_kws)

    fake = fake[..., field].copy()
    train = train[..., field].copy()

    if alpha_vlim:
        vmin = vmin or np.nanquantile(np.concatenate([fake, train]), alpha)
        vmax = vmax or np.nanquantile(np.concatenate([fake, train]), 1-alpha)
    else:
        vmin = vmin or min(np.nanmin(fake), np.nanmin(train))
        vmax = vmax or max(np.nanmax(fake), np.nanmax(train))

    nrows = 4
    ncols = 8
    total = nrows * ncols
    midpoint = total // 2
    midrow  = nrows // 2

    fig, axs, cax = makegrid(nrows, ncols, cbar_width=cbar_width, figsize=0.5)

    if vmax > 1.0:
        ndecimals = 0
    else:
        ndecimals = 2

    for i, ax in enumerate(axs.flat):
        if i < midpoint:
            pos = ax.get_position()
            im = ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])
            contourmap(fake[i, ...], ax=ax, vmin=vmin, vmax=vmax,
                       cmap=cmap, linewidth=linewidth, ndecimals=ndecimals)
        if i >= midpoint:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height])
            j = i - midpoint
            im = contourmap(train[j, ...], ax=ax, vmin=vmin, vmax=vmax,
                            cmap=cmap, linewidth=linewidth, ndecimals=ndecimals)

    axs[0, 0].set_ylabel("HazGAN")
    axs[midrow, 0].set_ylabel("ERA5")
    fig.colorbar(im, cax=cax, label=cbar_label, format=cbar_formatter)

    return fig
