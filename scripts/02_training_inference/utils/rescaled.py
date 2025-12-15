import os
from glob import glob
import numpy as np
from PIL import Image
import xarray as xr
from tqdm import tqdm

from hazGAN import statistics

EPS   = 1e-6


def λ(number_extremes:int, total_years:int, unit:int=1) -> float:
    """Calculate the extreme event rate for a given return period."""
    return number_extremes / (total_years / unit)


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


def yflip(array: np.ndarray, ax=1) -> np.ndarray:
    if array is not None:
        return np.flip(array, axis=ax)


def sample_dep(n, h, w, c, freq):
    rp_dep = np.logspace(0, 3, num=n, base=10)
    sf_dep = 1 / (freq * rp_dep)
    u_dep  = 1 - sf_dep
    u_dep  = np.repeat(u_dep, h*w*c, axis=0)
    u_dep  = u_dep.reshape(n, h, w, c)
    rp_dep = np.repeat(rp_dep, h*w*c, axis=0)
    rp_dep = rp_dep.reshape(n, h, w, c)
    return u_dep, rp_dep


def load_samples(
        png_dir, training_dir, threshold=15., ny=500, make_benchmarks=False) -> dict:
    """Load and process samples and training data for visualisation"""
    print(f"\nLoading samples from {png_dir} with domain=rescaled")
    # load all the png files
    png_list = glob(os.path.join(png_dir, "seed*.png"))
    png_list = sorted(png_list)
    print(f"Found {len(png_list)} PNG files in {png_dir}")
    print("WARNING: future work will use .npy to avoid quantisation issues.")

    samples = []
    for png in (pbar := tqdm(png_list, desc="Loading PNG files")):
        pbar.set_description(f"Loading {os.path.basename(png)}")
        with Image.open(png) as img:
            samples.append(np.array(img))
    samples = np.array(samples).astype(float)
    samples /= 255.
    print(f"Loaded {samples.shape} samples")

    # filepath changed Dec 2025
    stats_file = os.path.join(training_dir, "images", "rescaled", "image_stats.npz")
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
    u_trn = train.uniform.values
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

    # transformations (just ECDF here)
    y_trn = x_trn
    y_val = x_val
    print("Calculating generated ECDF")
    for c in range(3):
        print(f"\n Channel {c}")
        print(f"x_trn range: {x_trn[...,c].min()} to {x_trn[...,c].max()}")
        print(f"x_ref range: {x_ref[...,c].min()} to {x_ref[...,c].max()}")
        print(f"y_gen range: {y_gen[...,c].min()} to {y_gen[...,c].max()}")
        print(f"params: {params[...,c]}")

    u_gen = statistics.empiricalPIT(y_gen)

    # calculate how many of each set we need to generate n_y years of data
    nevents   = len(x_ref); print(f"{nevents=}")
    nextreme  = len(x_trn) + len(x_val); print(f"{nextreme=}")
    λ_storms  = λ(nevents, nyears)
    λ_extreme = λ(nextreme, nyears)
    nevents   = int(ny * λ_storms)
    nextremes = int(ny * λ_extreme)

    # make comparison samples for total dependence/independence assumptions
    n, h, w, c = u_gen.shape

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

    # get samples into data space (just y space here)
    x_gen = y_gen

    # reorder samples from largest to smallest max value
    gen_max = x_gen[..., 0].max(axis=(1,2))
    gen_ord = np.argsort(gen_max)[::-1]
    x_gen = x_gen[gen_ord]
    u_gen = u_gen[gen_ord]
    y_gen = y_gen[gen_ord]

    # make empirical copulas
    print("Calculating empirical copulas")
    cop_trn = statistics.empiricalPIT(x_trn)
    cop_val = statistics.empiricalPIT(x_val, x_trn)
    # only get copula for first 149 samples to match other sets
    nboot = n // nextreme
    copula_subsets = []
    for b in range(1, nboot):
        print(f"Calculating empirical copula for subset {b} of {nboot-1} size={nextreme}")
        start = b * nextreme
        end   = (b + 1) * nextreme
        copula_gen = statistics.empiricalPIT(x_gen[start:end, ...])
        copula_subsets.append(copula_gen)
    cop_gen = np.concatenate(copula_subsets, axis=0)

    # for southern hemisphere these should be upside-down in
    # heatmaps and correctly orientated in geographic plots 
    output_dict =  {
        'samples': {
            'u': yflip(u_gen),
            'y': yflip(y_gen),
            'x': yflip(x_gen),
            'mask': yflip(u_gen_mask),
            'copula': yflip(cop_gen)
        }, 'training': {
            'u': u_trn,
            'y': y_trn,
            'x': x_trn,
            'mask': u_trn_mask,
            'copula': cop_trn
        }, 'valid': {
            'u': u_val,
            'y': y_val,
            'x': x_val,
            'mask': u_val_mask,
            'copula': cop_val
        }
    }
    if make_benchmarks:
        print(f"Generating {nevents} (non-extreme) benchmark samples "\
            f"for {ny} years of data at rate {λ_storms:,.4f} storms/year")
    
        u_ind = np.random.uniform(1e-6, 1-1e-6, size=(nevents, h, w, c))
        u_dep, rp_dep = sample_dep(10, h, w, c, λ_storms)
        print("WARNING: ind/dep arrays may also need to flip lats.")
        x_ind = statistics.invPIT(u_ind, x_ref, params)
        x_dep = statistics.invPIT(u_dep, x_ref, params)
        benchmark_dict = {
            "independent": {
                'u': u_ind,
                'x': x_ind,
            }, "dependent": {
                'u': u_dep,
                'rp': rp_dep,
                'x': x_dep
            }  
        }
        output_dict = {**output_dict, **benchmark_dict}
    return output_dict
