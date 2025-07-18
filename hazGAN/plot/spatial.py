# %%
import numpy as np
import matplotlib.pyplot as plt

from ..statistics import pairwise_extremal_coeffs

from .base import CMAP
from .base import makegrid


def plot(fake, train, func, field=0, figsize=1., cmap=CMAP, vmin=None, vmax=None,
         title="Untitled", cbar_label=""):
    fake  = fake[..., field].copy()
    train = train[..., field].copy()
    
    fake_res  = func(fake)
    train_res = func(train)

    vmin = vmin or train_res.min()
    vmax = vmax or train_res.max()
    cmap = getattr(plt.cm, cmap)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, axs, cax = makegrid(1, 2, figsize=figsize, projection=None)

    im = axs[0].imshow(train_res, cmap=cmap)
    _  = axs[-1].imshow(fake_res, cmap=cmap)

    axs[0].set_title("Train", y=-0.1)
    axs[-1].set_title("Model", y=-0.1)
    fig.suptitle(title, y=1.05)

    fig.colorbar(im, cax=cax, label=cbar_label)

    return fig


def smith1990(array):
    array = array.astype(np.float16)
    return pairwise_extremal_coeffs(array)


def pearson(array):
    n, h, w = array.shape
    array = array.reshape(n, h * w)
    corrs = np.corrcoef(array.T)
    print(corrs.shape)
    return corrs


def taildependence(array, thresh=.9, metric='chi'):
    from hazGAN.R import R

    def f(a, b, thresh=thresh):
        return R.taildependence(a, b, thresh)[metric]

    n, h, w = array.shape
    array = array.reshape(n, h * w)

    chi_values = np.empty((h * w, h * w))
    for i in range(h * w):
        for j in range(i):
            print(i, j)
            chi = f(array[:, i], array[:, j])
            chi_values[i, j] = chi
            chi_values[j, i] = chi

    return chi_values


if __name__ == "__main__":
    import xarray as xr

    data_path = "/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/data.nc"
    data = xr.open_dataset(data_path)
    uniform = data['uniform'].values

    plot(uniform, uniform, smith1990, title="Extremal correlation", cbar_label="Χ") # too big 4096 * 4096
    plot(uniform, uniform, pearson, title="Pearson correlation", cbar_label="r")
    plot(uniform, uniform, taildependence, title="Tail dependence", cbar_label="χ")
# %%