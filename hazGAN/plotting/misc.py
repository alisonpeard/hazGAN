# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hazGAN.statistics import ecdf


__all__ = ["windvreturnperiod", "saffirsimpson_barchart"]


def maxwinds(x:np.ndarray, windchannel=0):
    return np.max(x[..., windchannel], axis=(1, 2)).squeeze()


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

            
def windvreturnperiod(x:np.ndarray, _lambda:float, ax=None, windchannel=0,
                transpose=False, **kwargs):
    """Plot wind speed against return period."""
    rp = returnperiod(x, _lambda, windchannel=windchannel)
    maxima = np.max(x[..., windchannel], axis=(1, 2)).squeeze()

    sorting = np.argsort(maxima)
    rp      = rp[sorting]
    maxima  = maxima[sorting]

    ax = ax or plt.gca()
    if transpose:
        ax.plot(maxima, rp, **kwargs)
        ax.set_xlabel("Wind Speed [m/s]")
        ax.set_ylabel("Return Period [years]")
    else:
        ax.plot(rp, maxima, **kwargs)
        ax.set_xlabel("Return Period [years]")
        ax.set_ylabel("Wind Speed [m/s]")

    # axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    ax.grid(True, linestyle='--', which='major', color='lightgrey', alpha=.2)
    ax.legend(loc='lower right', frameon=False)
    ax.set_xscale('log')

    return ax

        
def saffirsimpson_barchart(fake, train, title='',
                            xlabel="", yscale='linear',
                            barwidth=0.25,
                            scale="saffirsimpson", density=True,
                            figsize=(6, 3),
                            results={}) -> plt.Figure:
    """Plot bar charts comparing fake and train data across hurricane categories."""
    fake = maxwinds(fake)
    train = maxwinds(train)
    
    fake = pd.Series(fake.ravel())
    train = pd.Series(train.ravel())
    
    if scale == "saffirsimpson":
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
        cats = [17, 33, 43, 49, 58, 70, np.inf]
    elif scale == "fives":
        def category(x):
            if x < 15:
                return -1
            elif x < 20:
                return 0
            elif x < 25:
                return 1
            elif x < 30:
                return 2
            elif x < 35:
                return 3
            elif x < 40:
                return 4
            elif x < 45:
                return 5
            elif x < 50:
                return 6
            elif x < 55:
                return 7
            else:
                return 8
            
        cats = [15, 20, 25, 30, 35, 40, 45, 50, 55, np.inf]
            
    fake = fake.apply(category).astype(int)
    train = train.apply(category).astype(int)
    
    # Count frequencies in each category
    fake_counts = fake.value_counts().sort_index()
    train_counts = train.value_counts().sort_index()
    
    # Convert to probabilities/densities
    if density:
        fake_density = fake_counts / len(fake)
        train_density = train_counts / len(train)
    else:
        fake_density = fake_counts
        train_density = train_counts
    
    # Make sure all categories are represented (fill with zeros if missing)
    max_cat = max(fake_density.index.max(), train_density.index.max())

    all_categories = np.arange(-1, max_cat + 1)
    fake_density = pd.Series([fake_density.get(cat, 0) for cat in all_categories],
                            index=all_categories)
    train_density = pd.Series([train_density.get(cat, 0) for cat in all_categories],
                            index=all_categories)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Set positions of bars on x-axis
    r1 = np.arange(len(all_categories))
    r2 = [x + barwidth + 0.01 for x in r1]
    
    # Create bars
    ax.bar(r1, fake_density, width=barwidth, color="#DAE8FC", label='HazGAN',
        edgecolor='#6C8EBF', linewidth=0.5)
    ax.bar(r2, train_density, width=barwidth, color="#BAC8D3", label='ERA5',
        edgecolor='#23445D', linewidth=0.5)
    
    # Add extra details
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=.2)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.set_xticks([r + barwidth/2 for r in range(len(all_categories))])

    if scale == "saffirsimpson":
        ax.set_xticklabels(['Tropical\nDepression', 'Tropical\nStorm', 'Category\n1',
                            'Category\n2', 'Category\n3', 'Category\n4', 'Category\n5']
                            )
    elif scale == "fives":
        xticklabels = ['< 15 m/s', '15-20 m/s', '20-25 m/s', '25-30 m/s',
                            '30-35 m/s', '35-40 m/s', '40-45 m/s', '45-50 m/s',
                            '50-55 m/s', '> 55 m/s']
        ax.set_xticklabels(xticklabels[:len(all_categories)], rotation=30)
    ax.set_yscale(yscale)
    ax.legend()

    for catval, cat, fake, train in zip(cats, all_categories, fake_density, train_density):
        results[f"cat {cat}"] = {}
        results[f"cat {cat}"]["value"] = catval
        results[f"cat {cat}"]["fake"] = f"{fake:.2%}"
        results[f"cat {cat}"]["train"] = f"{train:.2%}"
        results[f"cat {cat}"]["difference"] = f"{abs(fake - train):.2%}"
        results[f"cat {cat}"]["%difference"] = f"{(abs(fake - train) / (train + 1e-6) * 100):.2f}%"
    
    return fig, ax


# def histogram(fake, train, func, title='Untitled', xlabel="x", yscale='linear',
#               bins=100, **func_kws) -> plt.Figure:
#     """Plot histograms of all pixel values."""
#     fake  = func(fake, **func_kws)
#     train = func(train, **func_kws)

#     fig, ax = plt.subplots(figsize=(8, 3))

#     xmin = min(np.nanmin(fake), np.nanmin(train))
#     xmax = max(np.nanmax(fake), np.nanmax(train))

#     if isinstance(bins, int):
#         bins = np.linspace(xmin, xmax, bins)
#     else: # if specific bins are given, use them as is
#         ax.set_xticks(bins)
#     ax.hist(fake, bins=bins, density=True,
#             color=yellows[1], edgecolor='k', alpha=.6, label="Generated");
#     ax.hist(train, bins=bins, density=True,
#             color=yellows[2], edgecolor='k', alpha=.6, label="Training");
#     ax.set_yscale(yscale)
#     ax.set_ylabel("Density")
#     fig.legend(loc='center', fontsize=12)
#     fig.suptitle(title, fontsize=16)
#     ax.set_xlabel(xlabel, fontsize=12)
#     ax.set_facecolor(yellows[0])

#     return fig



# def ravel(x:np.ndarray):
#     return x.ravel()


# def invPITmaxima(fake_u, train_u, fake, train):
#     fake_u  = fake_u[..., 0].reshape(-1, 4096)
#     train_u = train_u[..., 0].reshape(-1, 4096)
#     fake    = fake[..., 0].reshape(-1, 4096)
#     train   = train[..., 0].reshape(-1, 4096)

#     fake_max  = np.argmax(fake_u, axis=1)[..., np.newaxis]
#     train_max = np.argmax(train_u, axis=1)[..., np.newaxis]

#     fake_u  = np.take_along_axis(fake_u, fake_max, axis=1).squeeze()
#     train_u = np.take_along_axis(train_u, train_max, axis=1).squeeze()
#     fake    = np.take_along_axis(fake, fake_max, axis=1).squeeze()
#     train   = np.take_along_axis(train, train_max, axis=1).squeeze()

#     fake_sorting = np.argsort(fake_u)
#     train_sorting = np.argsort(train_u)

#     fake_u  = fake_u[fake_sorting]
#     train_u = train_u[train_sorting]
#     fake   = fake[fake_sorting]
#     train  = train[train_sorting]

#     fig, ax = plt.subplots(1, 2, figsize=(8, 3))
#     ax[0].plot(train_u, train, '-o', alpha=.5, markersize=5., label="Training")
#     ax[1].plot(fake_u, fake, '-o', alpha=.5, markersize=5., label="Generated")
#     fig.suptitle("invPIT of wind maxima", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()