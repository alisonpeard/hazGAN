# %%
import numpy as np
import matplotlib.pyplot as plt

from .base import makegrid
from .base import contourmap
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

    axs[0].set_title("Train", y=-0.1)
    axs[-1].set_title("Model", y=-0.1)

    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.suptitle(title, y=1.05)

    return fig


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


def taildependence(array, thresh=.9, metric='chi'):
    from hazGAN.R import R

    def calculate(u, thresh=thresh, metric=metric):
        n, h, w, c = u.shape
        u = u.reshape(n, h * w, c)

        def f(x, thresh=thresh):
            return R.taildependence(x[..., 0], x[..., 1], thresh)[metric]
        
        chi_values = []
        for i in range(h * w):
            chi = f(u[:, i, :])
            chi_values.append(chi)
        chi_values = np.stack(chi_values, axis=0)
        chi_values = chi_values.reshape(h, w)

        return chi_values
    return calculate(array, thresh, metric)


if __name__ == "__main__":
    import xarray as xr

    data_path = "/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/data.nc"
    data = xr.open_dataset(data_path)
    uniform = data['uniform'].values

    plot(uniform, uniform, taildependence, title="Tail dependence", cbar_label="Χ", figsize=.8);
    plot(uniform, uniform, pearson, vmin=-1, vmax=1, title="Pearson correlation", cbar_label="r", figsize=.8);
    plot(uniform, uniform, smith1990, title="Extremal coefficient", cbar_label="Χ", figsize=.8);
    

# %%
