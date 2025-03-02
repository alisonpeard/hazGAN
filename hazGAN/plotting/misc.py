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


def saffirsimpson_barchart(fake, train, title='Saffir-Simpson Scale',
                            xlabel="Category", yscale='linear', bar_width=0.35):
    """Plot bar charts comparing fake and train data across hurricane categories."""
    fake = maxwinds(fake)
    train = maxwinds(train)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    fake = pd.Series(fake.flatten())
    train = pd.Series(train.flatten())
    
    def category(x):
        if x < 17:
            return -1
        elif x < 33:
            return 0
        elif x < 43:
            return 1
        elif x < 49:
            return 2
        elif x < 58:
            return 3
        elif x < 70:
            return 4
        else:
            return 5
    
    fake = fake.apply(category).astype(int)
    train = train.apply(category).astype(int)
    
    # Count frequencies in each category
    fake_counts = fake.value_counts().sort_index()
    train_counts = train.value_counts().sort_index()
    
    # Convert to probabilities/densities
    fake_density = fake_counts / len(fake)
    train_density = train_counts / len(train)
    
    # Make sure all categories are represented (fill with zeros if missing)
    all_categories = np.arange(-1, 6)
    fake_density = pd.Series([fake_density.get(cat, 0) for cat in all_categories], index=all_categories)
    train_density = pd.Series([train_density.get(cat, 0) for cat in all_categories], index=all_categories)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set positions of bars on x-axis
    r1 = np.arange(len(all_categories))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    ax.bar(r1, fake_density, width=bar_width, color='#75C26A', label='Generated', edgecolor='black', linewidth=0.5)
    ax.bar(r2, train_density, width=bar_width, color='#5D6FC4', label='Training', edgecolor='black', linewidth=0.5)
    
    # Add extra details
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([r + bar_width/2 for r in range(len(all_categories))])
    ax.set_xticklabels(['Tropical\nDepression', 'Tropical\nStorm', 'Cat. 1',
                        'Cat. 2', 'Cat. 3', 'Cat. 4', 'Cat. 5'])
    ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    
    return fig


def histogram(fake, train, func, title='Untitled', xlabel="x", yscale='linear',
              bins=100, **func_kws) -> plt.Figure:
    """Plot histograms of all pixel values."""
    fake  = func(fake, **func_kws)
    train = func(train, **func_kws)

    fig, ax = plt.subplots(figsize=(8, 3))

    xmin = min(np.nanmin(fake), np.nanmin(train))
    xmax = max(np.nanmax(fake), np.nanmax(train))

    if isinstance(bins, int):
        bins = np.linspace(xmin, xmax, bins)
    else: # if specific bins are given, use them as is
        ax.set_xticks(bins)
    ax.hist(fake, bins=bins, density=True,
            color='C2', edgecolor='k', alpha=.6, label="Generated");
    ax.hist(train, bins=bins, density=True,
            color='blue', edgecolor='k', alpha=.6, label="Training");
    ax.set_yscale(yscale)
    ax.set_ylabel("Density")
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