import os
from glob import glob
import numpy as np
from PIL import Image
import xarray as xr
from tqdm import tqdm

from hazGAN import statistics

from pathlib import Path

eps   = 1e-6

def λ(number_extremes:int, total_years:int, unit:int=1) -> float:
    """Calculate the extreme event rate for a given return period."""
    return number_extremes / (total_years / unit)


def undo_scaling(x:np.array, stats:dict) -> np.array:
    mins = stats['min']
    maxs = stats['max']
    method = stats['method']
    arg = stats['param']
    ranges = maxs - mins

    if method == "rp":
        return x * ranges + mins
    elif method == "minmax":
        return(x / arg) * ranges + mins


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
        npy_dir, training_dir,
        threshold=15., nyrs=500, scaling="rp10000",
        make_benchmarks=False, bootstrap_copulas=True) -> dict:

    # load all the generated samples
    npy_list = glob(os.path.join(npy_dir, "seed*.npy"))
    npy_list = sorted(npy_list)

    gen = []
    for npy in (pbar := tqdm(npy_list, desc="Loading NPY files")):
        pbar.set_description(f"Loading {os.path.basename(npy)}")
        arr = np.load(npy)
        gen.append(arr)
    gen = np.array(gen).astype(float)
    gen /= 255.

    # undo (0,1) scaling
    stats_file = Path(training_dir) / "images" / scaling / 'rescaled' / "image_stats.npz"
    stats = np.load(stats_file)
    y_gen = undo_scaling(gen, stats)

    # sort the samples increasing by wind speed
    y_max = y_gen[..., 0].max(axis=(1, 2))
    y_ord  = np.argsort(y_max)
    y_gen  = y_gen[y_ord]

    # load training data for reference
    train   = xr.open_dataset(Path(training_dir) / "data.nc")
    nyears = int(train['time.year'].max() - train['time.year'].min() + 1)
    train['maxwind'] = train.sel(field='u10')['anomaly'].max(dim=['lat', 'lon'])
    train  = train.sortby('maxwind', ascending=False)

    if threshold:
        # warning: will affect ecdf values
        ref  = train.copy()
        mask = train.where(train['maxwind'] >= threshold, drop=True).time
        train = train.sel(time=mask)
        print(f"train ≥ {threshold}: {len(train.time)}.")
    else:
        ref = train.copy()
    
    # create arrays
    x_ref = ref.anomaly.values
    x_trn  = train.anomaly.values
    u_trn  = train.uniform.values
    params = train.params.values
    print(f"{params.shape=}")
    
    # check u_trn in [0, 1)
    if np.nanmax(u_trn) >= 1.:
        print("warning: some u_trn ≥ 1 (data.nc), rescaling.")
        u_trn_mask = (u_trn >= 1).astype(bool)
        u_trn *= (1-eps)
    else:
        u_trn_mask = None

    # make uniform (u) and reduced variate (y)
    y_trn = x_trn
    u_gen = statistics.empiricalPIT(y_gen)

    # calculate number of samples to use
    ntrn = len(x_trn)
    λ_trn = λ(ntrn, nyears)
    ngen = int(nyrs * λ_trn)
    n, h, w, c = u_gen.shape

    # randomly sample ngen samples
    print(f"taking {ngen} of {n} samples for {nyrs} years.")
    idx = np.random.choice(n, ngen, replace=False)
    u_gen = u_gen[idx]
    y_gen = y_gen[idx]

    # remove u=1 values
    if np.nanmax(u_gen) >= 1.:
        print("warning: some u_gen ≥ 1 (generated samples), rescaling.")
        u_gen_mask = (u_gen >= 1).astype(bool)
        u_gen *= (1-eps)
    else:
        u_gen_mask = None

    if np.nanmin(u_gen) <= 0.:
        print("warning: some u_gen ≤ 0, clipping.")
        u_gen = np.clip(u_gen, eps, float('inf'))

    # transform u_gen into data space
    x_gen = y_gen

    # reorder x_gen
    gen_max = x_gen[..., 0].max(axis=(1,2))
    gen_ord = np.argsort(gen_max)[::-1]
    x_gen = x_gen[gen_ord]
    u_gen = u_gen[gen_ord]
    y_gen = y_gen[gen_ord]

    # calculate empirical copulas
    print("calculating empirical copulas for train.")
    copula_trn = statistics.empiricalPIT(x_trn)

    # only get copula for first 150 samples to match trn
    if bootstrap_copulas:
        nboot = n // ntrn
        copula_subsets = []
        print(f"bootstrapping {nboot} empirical copulas of {ntrn} samples each.")
        for b in (pbar := tqdm(range(1, nboot), leave=False)):
            pbar.set_postfix_str(f"{b}/{nboot-1}")
            start = b * ntrn
            end   = (b + 1) * ntrn
            copula_gen = statistics.empiricalPIT(x_gen[start:end, ...])
            copula_subsets.append(copula_gen)
        copula_gen = np.concatenate(copula_subsets, axis=0)
    else:
        print(f"calculating empirical copulas for gen (no bootstrap).")
        copula_gen = statistics.empiricalPIT(x_gen, x_trn)

    # for southern hemisphere these should appear upside-down
    # in numpy but correctly orientated in geographic coords
    output_dict = {
        'samples': {
            'u': yflip(u_gen),
            'y': yflip(y_gen),
            'x': yflip(x_gen),
            'mask': yflip(u_gen_mask),
            'copula': yflip(copula_gen)
        }, 'training': {
            'u': u_trn,
            'y': y_trn,
            'x': x_trn,
            'mask': u_trn_mask,
            'copula': copula_trn
        }
    }
    if not make_benchmarks:
        print("not generating benchmark samples.")
        return output_dict
    else:
        # add total dependence / independence benchmarks
        nevents   = len(x_ref); print(f"{nevents=}")
        λ_events  = λ(nevents, nyears)
        nevents   = int(nyrs * λ_events)

        print(f"generating {nevents} (non-extreme) benchmark samples "\
            f"for {nyrs} years of data at rate {λ_events:,.4f} storms/year.")
        u_ind = np.random.uniform(1e-6, 1-1e-6, size=(nevents, h, w, c))
        u_dep, rp_dep = sample_dep(10, h, w, c, λ_events)
        print("warning: ind/dep arrays may also need to flip lats.")
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
