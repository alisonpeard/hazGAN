# %%
import os
from environs import Env
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

from hazGAN import statistics


domains = ["rescaled", "gaussian"] #["uniform", "gumbel", "gaussian"]
rescale_funcs = ["rp"]
rescale_args = [10_000]
output_fmt = "npy"
u10_threshold = 15


def save_stats_csv(path, stats_dict):
    with open(path, "w") as f:
        # header
        headers = ["metric"] + list(stats_dict.keys())
        headers = [h for h in headers]
        f.write(",".join(headers) + "\n")
        # rows
        metrics = list(next(iter(stats_dict.values())).keys())
        for metric in metrics:
            row = [metric]
            for section in stats_dict.keys():
                value = stats_dict[section][metric]
                val_str = f"{value:.4f}" if isinstance(value, (float, np.float64)) else str(value)
                row.append(val_str)
            f.write(",".join(row) + "\n")


def rescale_array(array, method, arg, domain):
    """Rescale array according to specified method."""
    array = np.copy(array)
    if method == "minmax":
        array_min = np.min(array, axis=(0, 1, 2), keepdims=True)
        array_max = np.max(array, axis=(0, 1, 2), keepdims=True)
        array = (array - array_min) / (array_max - array_min)
        array  = arg * array
        stats = {'min': array_min, 'max': array_max, 'param': arg,  'method': method}
        return array, stats

    elif method == "rp":
        if domain == "rescaled":
            n = array.shape[0]
            obs_max = array.max(axis=(0, 1, 2))
            k = np.log(arg) / np.log(n)
            array_max = k * obs_max
            array_min = array.min(axis=(0, 1, 2))
            array = (array - array_min) / (array_max - array_min)
            stats = {'min': array_min, 'max': array_max, 'param': arg, 'method': method}
            return array, stats

        ppf = getattr(statistics, domain)
        array_max = ppf(1 - 1 / arg)
        array_min = ppf(1 / arg)

        assert array_max > array.max(), \
            f"Return level max less than data max {array_max:.4f} < {array.max():.4f}"
        array = (array - array_min) / (array_max - array_min)
        stats = {'min': array_min, 'max': array_max, 'param': arg, 'method': method}
        return array, stats

    else:
        raise ValueError(f"unknown rescaling method: {method}.")


def main(rescale_method, rescale_arg, domain, output_format):
    env = Env()
    env.read_env(recurse=True)
    traindir = Path(env.str("TRAINDIR"))

    results = {} # store additional stats

    print(f"Loading training data from {traindir}.\n")
    ds = xr.open_dataset(traindir / 'data.nc')
    ds['u10_max'] = ds.sel(field='u10').anomaly.max(dim=['lon', 'lat'])
    mask = (ds['u10_max'] > u10_threshold).values
    idx  = np.where(mask)[0]
    ds   = ds.isel(time=idx)

    print(f"count(max(u10) > {u10_threshold} m/s): {ds.time.size}.")
    outdir = traindir / 'images' / (rescale_method + str(rescale_arg)) / domain 
    stats_path = outdir / "image_stats.npz"
    os.makedirs(outdir / output_format, exist_ok=True)

    nimgs = ds.time.size

    if domain == "rescaled":
         u = ds.anomaly.values
         results["rescaled"] = {}
         results["rescaled"]["max"] = u.max(axis=(0, 1, 2))
         results["rescaled"]["min"] = u.min(axis=(0, 1, 2))
    else:
        u = ds.uniform.values
        results["uniform"] = {}
        results["uniform"]["max"] = u.max(axis=(0, 1, 2))
        results["uniform"]["min"] = u.min(axis=(0, 1, 2))

        if not (u.min() >= 0.0) & ((u.max() < 1.0)):
            raise ValueError("percentiles not in [0, 1).")
    
    u = np.flip(u, axis=1)

    assert u.shape[1:] == (64, 64, 3), f"unexpected shape: {u.shape}"

    ppf = getattr(statistics, domain)
    y = ppf(u)
    y, stats = rescale_array(y, method=rescale_method, arg=rescale_arg, domain=domain)
    np.savez(stats_path, **stats)

    results[rescale_func] = stats

    results["output"] = {}
    results["output"]["max"] = y.max(axis=(0, 1, 2))
    results["output"]["min"] = y.min(axis=(0, 1, 2))

    # save images
    storm_paths = []
    for i in range(nimgs):
        y_i = y[i]
        assert np.all((y_i >= 0.) & (y_i < 1.)), \
            f"Array values out of [0,1) range: min {y_i.min()}, max {y_i.max()}"
        if output_format == "png":
            y_i = np.uint8(y_i * 255)
            img = Image.fromarray(y_i, 'RGB')
            output_path = outdir / output_format / f"storm_{i}.{output_format}"
            img.save(output_path)
        elif output_format == "npy":
            output_path = outdir / output_format / f"storm_{i}.{output_format}"
            np.save(output_path, y_i * 255)
        storm_paths.append(output_path)
    
    print(f"\nSaved {len(storm_paths)} images to {outdir}")

    # save stats csv
    resultspath = outdir / "image_stats.csv"
    print(f"\n{results}\n")
    save_stats_csv(resultspath, results)
    print(f"Saved results stats to {resultspath}")

    # zip files for transfer to remote training server
    zipdir = traindir / 'zipfiles' / (rescale_method + str(rescale_arg)) / domain
    os.makedirs(zipdir, exist_ok=True)
    zippath = zipdir / (output_format + ".zip")
    print(f"Zipping images to {zippath} ...")
    os.system(f"cd {outdir} && zip -r {zippath} . -i '*.{output_format}'")
    print("Done.")


if __name__ == "__main__":
    for rescale_func in rescale_funcs:
        for rescale_arg in rescale_args:
            for domain in domains:
                    main(rescale_func, rescale_arg, domain, output_fmt)
# %% 
