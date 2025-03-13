# %%
import numpy as np
from .base import CMAP
from .base import makegrid, contourmap, scalebar
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
        cbar_label='', cbar_width=0.2, linewidth=.1, alpha=1e-4, alpha_vlim=True, 
        nrows=4, ncols=8, ndecimals=1, **transform_kws):
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

    fig, axs, cax = makegrid(nrows, ncols, cbar_width=cbar_width, figsize=1.)
    for i, ax in enumerate(axs.flat):
        if i < midpoint:
            contourmap(fake[i, ...], ax=ax, vmin=vmin, vmax=vmax,
                       cmap=cmap, linewidth=linewidth, ndecimals=ndecimals)
        if i >= midpoint:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height])
            j = i - midpoint
            im = contourmap(train[j, ...], ax=ax, vmin=vmin, vmax=vmax,
                            cmap=cmap, linewidth=linewidth, ndecimals=ndecimals)

    # add (a) and (b) labels to top left of both blocks
    axs[0, 0].text(.2, .725, "(a)", transform=axs[0, 0].transAxes, ha='center', va='bottom',
                fontsize=20, 
                bbox=dict(facecolor='white', alpha=.8, linewidth=0, edgecolor='white', boxstyle='round,pad=0.2'))
    
    axs[midrow, 0].text(.2, .725, "(b)", transform=axs[midrow, 0].transAxes, ha='center', va='bottom',
                fontsize=20, 
                bbox=dict(facecolor='white', alpha=.8, linewidth=0, edgecolor='white', boxstyle='round,pad=0.2'))

    # add a scale bar
    scalebar(axs[-1, -1])
    scalebar(axs[midrow-1, -1])

    fig.colorbar(im, cax=cax, label=cbar_label)
    return fig


if __name__ == "__main__":
    import xarray as xr

    data_path = "/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/data.nc"
    data = xr.open_dataset(data_path)
    uniform = data['uniform'].values
    x       = data['anomaly'].values
    params  = data['params'].values

    uniform = np.nan_to_num(uniform, nan=1e-6, posinf=1 - 1e-6, neginf=1e-6)
    uniform = uniform.clip(1e-6, 1 - 1e-6)

    plot(uniform, uniform, title="Uniform samples");
    plot(uniform, uniform, transform=gumbel, title="Gumbel samples");
    plot(uniform, uniform, transform=anomaly, title="Anomaly samples", reference=x, params=params);

# %%