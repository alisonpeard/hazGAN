
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
# DOMAIN = ["gumbel", "gaussian", "uniform"][1]
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


def rescale(samples:np.array, stats_file:str) -> np.array:
    """Rescale samples from [0, 1] to data space using stored image statistics."""
    stats = np.load(stats_file)
    image_minima = stats['min']
    image_maxima = stats['max']
    n            = stats['n']
    image_range  = image_maxima - image_minima

    print(f"\nLoaded image statistics with shape {image_minima.shape}")
    print(f"Statistics: {image_minima.squeeze()} to {image_maxima.squeeze()} over {n.squeeze()} samples")

    print("WARNING: Current rescaling (Jan 2025) will be changed in future work.")
    samples = (samples * (n + 1) - 1) / (n - 1) * image_range + image_minima
    return samples


def sample_dep(n, h, w, c, freq):
    rp_dep = np.logspace(0, 3, num=n, base=10)
    sf_dep = 1 / (freq * rp_dep)
    u_dep  = 1 - sf_dep
    u_dep  = np.repeat(u_dep, h*w*c, axis=0)
    u_dep  = u_dep.reshape(n, h, w, c)
    rp_dep = np.repeat(rp_dep, h*w*c, axis=0)
    rp_dep = rp_dep.reshape(n, h, w, c)
    return u_dep, rp_dep


def load_samples(png_dir, training_dir, threshold=15., ny=500, domain="uniform"):
    """Load and process samples and training data for visualisation"""
    print(f"\nLoading samples from {png_dir} with {domain=}")
    # load all the png files
    png_list = glob(os.path.join(png_dir, "seed*.png"))
    png_list = sorted(png_list)
    print(f"Found {len(png_list)} PNG files in {png_dir}")
    print("WARNING: future work will use .npy to avoid quantisation issues.")

    samples = []
    for png in png_list:
        img = Image.open(png)
        samples.append(np.array(img))
    samples = np.array(samples).astype(float)
    samples /= 255.
    print(f"Loaded {samples.shape} samples")

    # filepath changed Dec 2025
    stats_file = os.path.join(training_dir, "images", domain, "image_stats.npz")
    y_gen = rescale(samples, stats_file)

    # order samples
    y_max = y_gen[..., 0].max(axis=(1, 2))
    y_ord  = np.argsort(y_max)
    y_gen  = y_gen[y_ord]

    # load training data for reference
    data   = xr.open_dataset(os.path.join(training_dir, "data.nc"))
    nyears = len(np.unique(data['time.year'].values))
    data['maxwind'] = data.sel(field='u10')['anomaly'].max(dim=['lat', 'lon'])
    trainmask = data.where(data['time.year'] != 2021, drop=True).time
    validmask = data.where(data['time.year'] == 2021, drop=True).time
    train  = data.sel(time=trainmask)
    valid = data.sel(time=validmask)
    del data

    if threshold:
        # this affects ECDF calculations so be careful
        print(f"\nApplying threshold of {threshold} m/s")
        ref  = train.copy()
        mask = train.where(train['maxwind'] > threshold, drop=True).time
        train = train.sel(time=mask)
        print(f"Selected {len(train.time)} training samples above threshold")

        mask = valid.where(valid['maxwind'] >= threshold, drop=True).time
        valid = valid.sel(time=mask)
        print(f"Selected {len(valid.time)} validation samples above threshold")
    else:
        ref = train.copy()
    
    x_ref = ref.anomaly.values

    train  = train.sortby('maxwind', ascending=False)
    x_trn  = train.anomaly.values
    u_trn  = train.uniform.values
    params = train.params.values
    print(f"Loaded {params.shape} parameters")
    
    valid = valid.sortby('maxwind', ascending=False)
    x_val = valid.anomaly.values
    u_val = valid.uniform.values

    # order training samples by x value
    x_max = x_trn[..., 0].max(axis=(1, 2))
    x_ord = np.argsort(x_max)[::-1]
    u_trn = u_trn[x_ord]
    x_trn = x_trn[x_ord]

    if (u_trn >= 1).sum() > 0:
        print("Some data.nc is greater than 1")
        u_trn_mask = (u_trn >= 1).astype(bool)
        u_trn *= (1-EPS)
    else:
        u_trn_mask = None

    if (u_val >= 1.).sum() > 0:
        print("Some valid data.nc is greater than 1")
        u_val_mask = (u_val >= 1).astype(bool)
        u_val *= (1-EPS)
    else:
        u_val_mask = None

    # transformations
    ppf = getattr(statistics, domain)
    cdf = getattr(statistics, "inv_" + domain)
    y_trn = ppf(u_trn)
    y_val = ppf(u_val)
    u_gen = cdf(y_gen)

    # calculate how many of each set we need to generate n_y years of data
    nevents   = len(x_ref); print(f"{nevents=}")
    nextreme  = len(x_trn) + len(x_val); print(f"{nextreme=}")
    λ_storms  = λ(nevents, nyears)
    λ_extreme = λ(nextreme, nyears)
    nevents   = int(ny * λ_storms)
    nextremes = int(ny * λ_extreme)
    nhazmaps  = 10
    print(f"Generating {nevents} (non-extreme) benchmark samples "\
          f"for {ny} years of data at rate {λ_storms:,.4f} storms/year")

    # make comparison samples for total dependence/independence assumptions
    n, h, w, c = u_gen.shape
    u_ind = np.random.uniform(1e-6, 1-1e-6, size=(nevents, h, w, c))
    u_dep, rp_dep = sample_dep(nhazmaps, h, w, c, λ_storms)

    # randomly sample nextremes from the sampled data
    print(f"Sampling {nextremes} samples from {n} synthetic events.")
    idx = np.random.choice(n, nextremes, replace=False)
    u_gen = u_gen[idx]
    y_gen = y_gen[idx]

    # remove [0,1] values
    if (u_gen >= 1).sum() > 0:
        print("WARNING: Some uniform samples are greater than 1")
        u_gen_mask = (u_gen >= 1).astype(bool)
        u_gen *= (1 - EPS)
    else:
        u_gen_mask = None

    if np.nanmin(u_gen) <= 0.:
        print("WARNING: Some uniform samples == 0")
        u_gen = np.clip(u_gen, EPS, float('inf'))

    # get samples into data space
    u_tmp = np.flip(u_gen, axis=1)
    x_gen = statistics.invPIT(u_tmp, x_ref, params)
    x_gen = np.flip(x_gen, axis=1)
    del u_tmp

    print("WARNING: ind/dep arrays may also need to flip lats.")
    x_ind = statistics.invPIT(u_ind, x_ref, params)
    x_dep = statistics.invPIT(u_dep, x_ref, params)

    # reorder samples in x space
    gen_max = x_gen[..., 0].max(axis=(1,2))
    gen_ord = np.argsort(gen_max)[::-1]
    x_gen = x_gen[gen_ord]
    u_gen = u_gen[gen_ord]
    y_gen = y_gen[gen_ord]

    # # negate MSLP --> no. do after medians restored
    # x_gen[..., 2] *= -1
    # x_trn[..., 2] *= -1
    # x_val[..., 2] *= -1
    # x_ind[..., 2] *= -1
    # x_dep[..., 2] *= -1

    # for southern hemisphere these should be upside-down in
    # heatmaps and correctly orientated in geographic plots 
    return {
        'samples': {
            'u': yflip(u_gen),
            'y': yflip(y_gen),
            'x': yflip(x_gen),
            'mask': yflip(u_gen_mask)
        }, 'training': {
            'u': u_trn,
            'y': y_trn,
            'x': x_trn,
            'mask': u_trn_mask
        }, 'valid': {
            'u': u_val,
            'y': y_val,
            'x': x_val,
            'mask': u_val_mask
        }, "independent": {
            'u': u_ind,
            'x': x_ind,
        }, "dependent": {
            'u': u_dep,
            'rp': rp_dep,
            'x': x_dep
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