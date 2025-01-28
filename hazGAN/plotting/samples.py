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
         cbar_label='', cbar_width=0.2, linewidth=.1, alpha=1e-4, **transform_kws):
    """Plot training samples on top row and generated samples on bottom row."""

    transform = transform or identity
    fake  = transform(fake, **transform_kws)
    train = transform(train, **transform_kws)

    fake = fake[..., field].copy()
    train = train[..., field].copy()

    def sort_by_wind(array):
        maxima = np.max(array, axis=(1, 2))
        sorting = np.argsort(maxima)[::-1]
        return array[sorting]

    fake = sort_by_wind(fake)
    train = sort_by_wind(train)

    # vmin = vmin or min(np.nanmin(fake), np.nanmin(train))
    # vmax = vmax or max(np.nanmax(fake), np.nanmax(train))
    vmin = vmin or np.nanquantile(np.concatenate([fake, train]), alpha)
    vmax = vmax or np.nanquantile(np.concatenate([fake, train]), 1-alpha)

    fig, axs, cax = makegrid(8, 8, cbar_width=cbar_width, figsize=1.2)
    for i, ax in enumerate(axs.flat):
        if i < 32:
            contourmap(train[i, ...], ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=linewidth)
        if i >= 32:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height])
            j = i - 32
            im = contourmap(fake[j, ...], ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=linewidth)

    # add A and B labels to top left of both blocks
    axs[0, 0].text(.25, .6, "A", transform=axs[0, 0].transAxes, ha='center', va='bottom', fontsize=22, weight='bold')
    axs[4, 0].text(.25, .6, "B", transform=axs[4, 0].transAxes, ha='center', va='bottom', fontsize=22, weight='bold')

    # add a scale bar
    scalebar(axs[-1, -1])
    scalebar(axs[3, -1])

    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.suptitle(title, y=1.02, fontsize=22)

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