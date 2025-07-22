# %%
import numpy as np
import matplotlib.pyplot as plt

from hazGAN.statistics import ecdf

yellows = ['#fff2ccff', '#f1c232ff', '#e69138ff', '#cc4125ff']
blues = ['#fff2ccff', '#a2c4c9ff', '#0097a7ff', '#0b5394ff']

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
    plt.style.use('seaborn-v0_8-whitegrid')
    rp = returnperiod(x, _lambda, windchannel=windchannel)
    maxima = np.max(x[..., windchannel], axis=(1, 2)).squeeze()

    sorting = np.argsort(maxima)
    rp      = rp[sorting]
    maxima  = maxima[sorting]

    ax = ax or plt.gca()
    if transpose:
        ax.plot(maxima, rp, **kwargs)
        ax.set_xlabel("Wind Speed [m/s]", fontsize=14, fontweight='bold')
        ax.set_ylabel("Return Period [years]", fontsize=14, fontweight='bold')
    else:
        ax.plot(rp, maxima, **kwargs)
        ax.set_xlabel("Return Period [years]", fontsize=14, fontweight='bold')
        ax.set_ylabel("Wind Speed [m/s]", fontsize=14, fontweight='bold')

    # axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=5)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xscale('log')
    
    # legend
    legend = ax.legend(loc='lower right', 
            frameon=True, 
            framealpha=0.9,
            edgecolor='gray',
            fontsize=12)
    legend.get_frame().set_linewidth(0)

    # background
    ax.set_facecolor('#F8F9FA')     

    return ax


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
        
def saffirsimpson_barchart(fake, train, title='',
                            xlabel="", yscale='linear',
                            bar_width=0.25, grid=True,
                            scale="saffirsimpson"):
    """Plot bar charts comparing fake and train data across hurricane categories."""
    fake = maxwinds(fake)
    train = maxwinds(train)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    fake = pd.Series(fake.flatten())
    train = pd.Series(train.flatten())
    
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

    
    fake = fake.apply(category).astype(int)
    train = train.apply(category).astype(int)
    
    # Count frequencies in each category
    fake_counts = fake.value_counts().sort_index()
    train_counts = train.value_counts().sort_index()
    
    # Convert to probabilities/densities
    fake_density = fake_counts / len(fake)
    train_density = train_counts / len(train)
    
    # Make sure all categories are represented (fill with zeros if missing)
    max_cat = max(fake_density.index.max(), train_density.index.max())
    print(f"Max category: {max_cat}")
    all_categories = np.arange(-1, max_cat + 1)
    fake_density = pd.Series([fake_density.get(cat, 0) for cat in all_categories],
                            index=all_categories)
    train_density = pd.Series([train_density.get(cat, 0) for cat in all_categories],
                            index=all_categories)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 5))

    # Set positions of bars on x-axis
    r1 = np.arange(len(all_categories))
    r2 = [x + bar_width + 0.01 for x in r1]
    
    # Create bars
    ax.bar(r1, fake_density, width=bar_width, color="#DAE8FC", label='HazGAN',
        edgecolor='#6C8EBF', linewidth=0.5)
    ax.bar(r2, train_density, width=bar_width, color="#BAC8D3", label='ERA5',
        edgecolor='#23445D', linewidth=0.5)
    
    # Add extra details
    if grid:
        ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Probability density", fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.set_xticks([r + bar_width/2 for r in range(len(all_categories))])

    if scale == "saffirsimpson":
        ax.set_xticklabels(['Tropical\nDepression', 'Tropical\nStorm', 'Category\n1',
                            'Category\n2', 'Category\n3', 'Category\n4', 'Category\n5'],
                            fontsize=18)
    elif scale == "fives":
        xticklabels = ['< 15 m/s', '15-20 m/s', '20-25 m/s', '25-30 m/s',
                            '30-35 m/s', '35-40 m/s', '40-45 m/s', '45-50 m/s',
                            '50-55 m/s', '> 55 m/s']
        ax.set_xticklabels(xticklabels[:len(all_categories)], fontsize=18)
    ax.set_yscale(yscale)
    ax.legend(
        fontsize=18,
    )
    fig.tight_layout()

    for cat, fake, train in zip(all_categories, fake_density, train_density):
        print(f"Cat {cat}: Fake: {fake:.2%}, Train: {train:.2%}, Difference: {abs(fake - train):.2%}, %Difference: {(abs(fake - train) / (train + 1e-6) * 100):.2f}%")
    
    return fig, ax


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
            color=yellows[1], edgecolor='k', alpha=.6, label="Generated");
    ax.hist(train, bins=bins, density=True,
            color=yellows[2], edgecolor='k', alpha=.6, label="Training");
    ax.set_yscale(yscale)
    ax.set_ylabel("Density")
    fig.legend(loc='center', fontsize=12)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_facecolor(yellows[0])

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