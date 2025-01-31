# %%
import numpy as np
import matplotlib.pyplot as plt

from hazGAN.statistics import ecdf


def invPITmaxima(fake_u, train_u, fake, train):
    fake_u  = fake_u[..., 0].reshape(-1, 4096)
    train_u = train_u[..., 0].reshape(-1, 4096)
    fake    = fake[..., 0].reshape(-1, 4096)
    train   = train[..., 0].reshape(-1, 4096)

    fake_max  = np.argmax(fake_u, axis=1)[..., np.newaxis]
    train_max = np.argmax(train_u, axis=1)[..., np.newaxis]

    fake_u  = np.take_along_axis(fake_u, fake_max, axis=1).squeeze()
    train_u = np.take_along_axis(train_u, train_max, axis=1).squeeze()
    fake    = np.take_along_axis(fake, fake_max, axis=1).squeeze()
    train   = np.take_along_axis(train, train_max, axis=1).squeeze()

    fake_sorting = np.argsort(fake_u)
    train_sorting = np.argsort(train_u)

    fake_u  = fake_u[fake_sorting]
    train_u = train_u[train_sorting]
    fake   = fake[fake_sorting]
    train  = train[train_sorting]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(train_u, train, '-o', alpha=.5, markersize=5., label="Training")
    ax[1].plot(fake_u, fake, '-o', alpha=.5, markersize=5., label="Generated")
    fig.suptitle("invPIT of wind maxima", fontsize=16)
    ax[0].legend()
    ax[1].legend()

            
def windvsreturnperiod(x:np.ndarray, _lambda:float, ax=None, windchannel=0,
                transpose=False, **kwargs):
    """Plot wind speed against return period."""
    rp = returnperiod(x, _lambda, windchannel=windchannel)
    maxima = np.max(x[..., windchannel], axis=(1, 2)).squeeze()

    sorting = np.argsort(maxima)
    rp      = rp[sorting]
    maxima  = maxima[sorting]

    ax = ax or plt.gca()
    if transpose:
        ax.plot(maxima, rp, alpha=.5, markersize=5., **kwargs)
        ax.set_xlabel("Wind Speed [m/s]")
        ax.set_ylabel("Return Period [years]")
    else:
        ax.plot(rp, maxima, alpha=.5, markersize=5., **kwargs)
        ax.set_xlabel("Return Period [years]")
        ax.set_ylabel("Wind Speed [m/s]")
            
    return ax


def histogram(fake, train, func, title='Untitled', xlabel="x", yscale='linear',
              **func_kws) -> plt.Figure:
    """Plot histograms of all pixel values."""
    fake = func(fake, **func_kws)
    train = func(train, **func_kws)

    fig, ax = plt.subplots(figsize=(8, 3))

    xmin = min(np.nanmin(fake), np.nanmin(train))
    xmax = max(np.nanmax(fake), np.nanmax(train))

    bins = np.linspace(xmin, xmax, 100)
    ax.hist(fake, bins=bins, density=True,
            color='C2', edgecolor='k', alpha=.6, label="Generated");
    ax.hist(train, bins=bins, density=True,
            color='blue', edgecolor='k', alpha=.8, label="Training");
    ax.set_yscale(yscale)
    fig.legend(loc='center', fontsize=12)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)

    return fig

def returnperiod(x:np.ndarray, _lambda:float, ax=None, windchannel=0):
    """Calculate return periods for a given dataset."""               
    x = x[..., windchannel].squeeze()
    maxima = np.max(x, axis=(1, 2))

    maxima = maxima.squeeze()
    assert len(maxima.shape) == 1
    cdf = ecdf(maxima)(maxima)
    svl = 1 - cdf
    rp  = 1 / (_lambda * svl)
    return rp

def ravel(x:np.ndarray):
    return x.ravel()

def maxwinds(x:np.ndarray, windchannel=0):
    return np.max(x[..., windchannel], axis=(1, 2)).squeeze()