
"""Module for loading, transforming, and plotting samples from the hazGAN model."""
import os
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr

from hazGAN import statistics

__all__ = ['plot', 'load_samples', 'yflip']


CMAP  = "Spectral_r"
DOMAIN = ["gumbel", "gaussian", "uniform"][1]
EPS   = 1e-6


def λ(number_extremes:int, total_years:int, unit:int=1) -> float:
    """Calculate the extreme event rate for a given return period."""
    return number_extremes / (total_years / unit)


def get_monthly_medians(traindir, month=None):
    medians = pd.read_csv(os.path.join(traindir, "medians.csv"))
    if month is None:
        medians = medians[['grid', 'u10', 'tp', 'mslp']].groupby('grid').mean().values
    else:
        medians = medians[medians['month'] == month][['grid', 'u10', 'tp', 'mslp']].groupby('grid').mean().values
    medians = medians.reshape(1, 64, 64, 3) # for broadcasting
    return yflip(medians)


def yflip(array: np.ndarray, ax=1) -> np.ndarray:
    if array is not None:
        return np.flip(array, axis=ax)


def load_samples(samples_dir, training_dir, model, threshold=None, ny=500, sampletype='samples'):
    """Load and process samples and training data for visualisation"""
    # load samples
    samples_path = os.path.join(samples_dir, model, "results", sampletype)
    samples_list = glob(os.path.join(samples_path, "seed*.png"))
    samples_list = sorted(samples_list)
    print(f"Found {len(samples_list)} samples in {samples_path}")

    samples = []
    for png in samples_list:
        img = Image.open(png)
        samples.append(np.array(img))
    samples = np.array(samples).astype(float)
    samples /= 255.
    print(f"Loaded {samples.shape} samples")

    # load gumbel scaling statistics
    stats_file = os.path.join(training_dir, "image_stats.npz")
    stats = np.load(stats_file)
    image_minima = stats['min']
    image_maxima = stats['max']
    n            = stats['n']
    image_range  = image_maxima - image_minima

    print(f"Loaded image statistics with shape {image_minima.shape}")

    # rescale images 
    Warning("Using new rescaling from Wed 22 January 2025")
    samples = (samples * (n + 1) - 1) / (n - 1) * image_range + image_minima

    # order samples
    sample_maxima = samples[..., 0].max(axis=(1,2))
    sample_order  = np.argsort(sample_maxima)#[::-1]
    samples       = samples[sample_order]
    # samples       = truncate(samples)

    # inspect sample distribution
    _, ax = plt.subplots(figsize=(6, 3))
    ax.hist(samples.ravel(), bins=50, color='lightgrey', edgecolor='k', density=True);
    ax.set_xlabel("Pixel value");
    ax.set_ylabel("Density");
    ax.set_title("Histogram of all reduced variate samples") # this should be uniform

    # load training data for reference
    data   = xr.open_dataset(os.path.join(training_dir, "data.nc"))
    nyears = len(np.unique(data['time.year'].values));Warning("Need more robust year counting")
    data['maxwind'] = data.sel(field='u10')['anomaly'].max(dim=['lat', 'lon'])
    trainmask = data.where(data['time.year'] != 2021, drop=True).time
    validmask = data.where(data['time.year'] == 2021, drop=True).time
    valid = data.sel(time=validmask)
    data  = data.sel(time=trainmask)

    if threshold:
        # this affects ECDF calculations so be careful
        print(f"Applying threshold of {threshold} mps")
        ref   = data.copy()
        tmask = data.where(data['maxwind'] >= 15., drop=True).time
        data  = data.sel(time=tmask)
    else:
        ref = data.copy()
    
    data   = data.sortby('maxwind', ascending=False)
    x      = data.anomaly.values
    x_ref  = ref.anomaly.values
    u      = data.uniform.values
    params = data.params.values
    print(f"Loaded {params.shape} parameters")

    if threshold is not None:
        tmask = valid.where(valid['maxwind'] >= 15., drop=True).time
        valid = valid.sel(time=tmask)
    
    valid = valid.sortby('maxwind', ascending=False)
    x_valid = valid.anomaly.values
    u_valid = valid.uniform.values

    # order training samples by x value
    x_maxima = x[..., 0].max(axis=(1, 2))
    x_order  = np.argsort(x_maxima)[::-1]
    u        = u[x_order]
    x        = x[x_order]

    if (u >= 1).sum() > 0:
        print("Some data.nc is greater than 1")
        invalid_umask = (u >= 1).astype(bool)
        u *= (1-EPS)
    else:
        invalid_umask = None

    if (u_valid >= 1.).sum() > 0:
        print("Some valid data.nc is greater than 1")
        invalid_valid_umask = (u_valid >= 1).astype(bool)
        u_valid *= (1-EPS)
    else:
        invalid_valid_umask = None

    # transformations
    ppf = getattr(statistics, DOMAIN)
    cdf = getattr(statistics, "inv_" + DOMAIN)
    x_gumbel        = ppf(u)
    valid_gumbel    = ppf(u_valid)
    samples_uniform = cdf(samples)

    # calculate how many of each set we need to generate n_y years of data
    nstorms   = len(x_ref)
    nextreme  = len(x) + len(x_valid)
    λ_storms  = λ(nstorms, nyears)
    λ_extreme = λ(nextreme, nyears)
    nstorms   = int(ny * λ_storms)
    nextremes = int(ny * λ_extreme)
    nhazmaps  = 10
    print(f"Generating {nstorms} (non-extreme) benchmark samples for {ny} years of data at rate {λ_storms:,.4f} storms/year")

    # make comparison samples for total dependence/independence assumptions
    n, h, w, c = samples_uniform.shape
    independent_uniform = np.random.uniform(1e-6, 1-1e-6, size=(nstorms, h, w, c))
    dependent_rps       = np.logspace(0, 3, num=nhazmaps, base=10)
    dependent_survival  = 1 / (λ_storms * dependent_rps)
    dependent_uniform   = 1 - dependent_survival
    dependent_uniform   = np.repeat(dependent_uniform, h*w*c, axis=0)
    dependent_uniform   = dependent_uniform.reshape(nhazmaps, h, w, c)
    dependent_rps       = np.repeat(dependent_rps, h*w*c, axis=0)
    dependent_rps       = dependent_rps.reshape(nhazmaps, h, w, c)

    # randomly sample nextremes from the sampled data
    idx = np.random.choice(n, nextremes, replace=False)
    samples_uniform = samples_uniform[idx]
    samples         = samples[idx]

    # remove [0,1] values
    if (samples_uniform >=1).sum() > 0:
        Warning("Some uniform samples are greater than 1")
        invalid_mask = (samples_uniform>=1).astype(bool)
        samples *= (1-EPS)
    else:
        invalid_mask = None

    if np.nanmin(samples_uniform) <= 0.:
        Warning("Some uniform samples == 0")
        samples_uniform = np.clip(samples_uniform, EPS, float('inf'))

    # get samples into data space
    samples_temp  = np.flip(samples_uniform, axis=1)
    samples_x     = statistics.invPIT(samples_temp, x_ref, params, margins=DOMAIN)
    samples_x     = np.flip(samples_x, axis=1)

    Warning("Check whether assumption arrays need to be flipped.")
    independent_x = statistics.invPIT(independent_uniform, x_ref, params, margins=DOMAIN)
    dependent_x   = statistics.invPIT(dependent_uniform, x_ref, params, margins=DOMAIN)
    del samples_temp

    # reorder samples in x space
    sample_maxima   = samples_x[..., 0].max(axis=(1,2))
    sample_order    = np.argsort(sample_maxima)[::-1]
    samples_x       = samples_x[sample_order]
    samples_uniform = samples_uniform[sample_order]
    samples         = samples[sample_order]

    # negate MSLP
    samples_x[..., 2] *= -1
    x[..., 2]         *= -1
    x_valid[..., 2]   *= -1
    independent_uniform[..., 2] *= -1

    # these will be upside-down in heatmaps and correctly orientated
    # in contour plots because y goes from 80 -> 95
    return {
        'samples': {
            'uniform': yflip(samples_uniform),
            'gumbel':  yflip(samples),
            'data':    yflip(samples_x),
            'mask':    yflip(invalid_mask)
        },
        'training': {
            'uniform': u, # yflip(u),
            'gumbel':  x_gumbel, # yflip(x_gumbel),
            'data':    x, # yflip(x),
            'mask':    invalid_umask #yflip(invalid_umask, 0)
        },
        'valid': {
            'uniform': u_valid, # yflip(u_valid),
            'gumbel':  valid_gumbel, # yflip(valid_gumbel),
            'data':    x_valid, # yflip(x_valid),
            'mask':    invalid_valid_umask # yflip(invalid_valid_umask, 0)
        },
        'assumptions': {
            'independent': independent_x,
            'dependent_u': dependent_uniform,
            'dependent_rp': dependent_rps,
            'dependent': dependent_x
        }        
    }


def plot(array, field, yflip=False, contours=False, mask=None, title='',
         exclude_mask=False, print_stats=True,
         standardise_colours=True,
         vmin=None, vmax=None,
         cmap=CMAP, levels=13):
    """

    """
    array = array.copy()
    if yflip:
        array = np.flip(array, axis=1)

    if exclude_mask and mask is not None:
        array[mask] = np.nan

    if standardise_colours:
        vmin = vmin or np.nanmin(array[..., field])
        vmax = vmax or np.nanmax(array[..., field])

    fig, axs = plt.subplots(8, 8, figsize=(16, 13),
                            sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0., 'wspace': 0.})
    for i, ax in enumerate(axs.flat):
        if contours:
            levels = np.linspace(vmin, vmax, levels)
            im = ax.contourf(array[i, ..., field],
            cmap=cmap,
            levels=levels
            )
        else:
            im = ax.imshow(array[i, ..., field],
                           cmap=cmap,
                           vmin=vmin, vmax=vmax
                           )
        if mask is not None:
            ax.contour(mask[i, ..., field],
                       colors='k',
                       linewidths=1
                       )
        ax.label_outer()
    
    if standardise_colours:
        fig.colorbar(im, ax=list(axs.flat))
    fig.suptitle(title, y=.9)

    if print_stats:
        print(f"\nStatistics for {title}:\n----------------------")
        print(f"Min: {np.nanmin(array[..., field]):.4f}")
        print(f"Max: {np.nanmax(array[..., field]):.4f}")
        print(f"Mean: {np.nanmean(array[..., field]):.4f}")
        print(f"Std: {np.nanstd(array[..., field]):.4f}")