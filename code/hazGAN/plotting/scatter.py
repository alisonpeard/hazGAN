#TODO: Tidy up !!
import random
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from .base import CMAP
from ..constants import channel_labels
from ..statistics import get_extremal_coeffs_nd


def plot(fake, real, field=0, pixels=None, cmap=CMAP, s=10,
         xlabel=None, ylabel=None, figsize=(6, 3)):
        _, h, w, c = real.shape
        if pixels is None:
            i, j = random.sample(range(h * w), 2)
        else:
            i, j = pixels

        fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

        single_scatter(real[..., field], ax=axs[0], sample_pixels=[j,i], cmap=cmap)
        single_scatter(fake[..., field], ax=axs[1], sample_pixels=[j,i], cmap=cmap)

        axs[0].set_title(f"Data (n={len(real)})", fontsize=13)
        axs[1].set_title(f"Samples (n={len(fake)})", fontsize=13)

        for ax in axs:
            if ylabel:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(f"pixel index {i}", fontsize=14, fontweight='bold')

            if xlabel:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(f"pixel index {j}", fontsize=14, fontweight='bold')

            # ax.set_facecolor('#f3f3f3ff')
            ax.label_outer()
        
        plt.tight_layout()

        fig.suptitle(channel_labels[field].capitalize(), y=1.05, fontsize=14, fontweight='bold')

        return fig


def single_scatter(data, ax=None, sample_pixels=None, cmap=CMAP, s=10):
    """Scatterplot for two variables, coloured by density."""
    h, w = data.shape[1:3]
    n = h * w

    if sample_pixels is None:
        sample_pixels_x = random.sample(range(n), 1)
        sample_pixels_y = random.sample(range(n), 1)
    else:
        assert sample_pixels[0] != sample_pixels[1]
        sample_pixels_x = [sample_pixels[0]]
        sample_pixels_y = [sample_pixels[1]]

    data_ravel = np.reshape(data, [len(data), n])

    sample_x = np.take(data_ravel, sample_pixels_x, axis=1)
    sample_y = np.take(data_ravel, sample_pixels_y, axis=1)

    axtitle = f"Pixels ({sample_pixels_x[0]}, {sample_pixels_y[0]})"

    if not isinstance(sample_x, np.ndarray):
        sample_x = sample_x.numpy()
        sample_y = sample_y.numpy()

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    scatter_density(sample_x, sample_y, ax, title=axtitle, cmap=cmap, s=s)


def scatter_density(x, y, ax, title='', cmap=CMAP, s=10):
    xy = np.hstack([x, y]).transpose()
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=s, cmap=cmap)
    ax.set_title(title)
    return ax
