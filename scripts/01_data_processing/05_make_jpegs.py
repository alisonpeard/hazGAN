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


# parameters
i, j, k = 3, 1, 0
DOMAINS = ["uniform", "gumbel", "gaussian"] # "rescaled"
RESCALE_METHODS = ["rp"] # minmax
RESCALE_ARGS = [10_000] # 0.9 for minmax
FORMATS = ["png"] # TODO: add npy from hazGAN2 later

# hardcode for now
WINDTHRESHOLD = 15


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
            raise ValueError("Return period scaling not defined for 'rescaled' domain")

        ppf = getattr(statistics, domain)
        array_max = ppf(1 - 1 / arg)
        array_min = ppf(1 / arg)

        assert array_max > array.max(), \
            f"Return level max less than data max {array_max:.4f} < {array.max():.4f}"
        array = (array - array_min) / (array_max - array_min)
        stats = {'min': array_min, 'max': array_max, 'param': arg, 'method': method}
        return array, stats



def main(rescale_method, rescale_arg, domain, output_format):
    env = Env()
    env.read_env(recurse=True)
    traindir = Path(env.str("TRAINDIR"))

    print(f"Loading training data from {traindir}")
    ds = xr.open_dataset(traindir / 'data.nc')
    ds['windmax'] = ds.sel(field='u10').anomaly.max(dim=['lon', 'lat'])
    mask = (ds['windmax'] > WINDTHRESHOLD).values
    idx  = np.where(mask)[0]
    ds   = ds.isel(time=idx)

    print(f"\nFound {ds.time.size} training events with maximum wind exceeding {WINDTHRESHOLD} m/s")

    outdir = traindir / 'images' / (rescale_method + str(rescale_arg)) / domain 
    stats_path = outdir / "image_stats.npz"
    os.makedirs(outdir / output_format, exist_ok=True)

    nimgs = ds.time.size

    if domain == "rescaled":
         u = ds.anomaly.values
    else:
        u = ds.uniform.values
        print(f"\nINFO: Maximum u-value found is {u.max():.6f}")
        print(f"INFO: Corresponds to {1/(1-u.max()):,.0f}-year return level assumption")
        print(f"INFO: Minimum u-value found is {u.min():.6f}")
        print(f"INFO: Corresponds to {1/(1-u.min()):,.0f}-year return level assumption")

        if not ((u.max() < 1.0) and (u.min() >= 0.0)):
            raise ValueError("Percentiles not in [0, 1) range")
    
    u = np.flip(u, axis=1)

    assert u.shape[1:] == (64, 64, 3), f"Unexpected shape: {u.shape}"

    ppf = getattr(statistics, domain)
    y = ppf(u)
    y, stats = rescale_array(y, method=rescale_method, arg=rescale_arg, domain=domain)
    np.savez(stats_path, **stats)

    # save images
    storm_paths = []
    for i in range(nimgs):
        y_i = y[i]
        assert np.all((y_i >= 0.) & (y_i < 1.)), \
            f"Array values out of [0,1) range: min {y_i.min()}, max {y_i.max()}"
        y_i = np.uint8(y_i * 255)
        img = Image.fromarray(y_i, 'RGB')
        output_path = outdir / output_format / f"storm_{i}.{output_format}"
        img.save(output_path)
        storm_paths.append(output_path)

    print(f"\nSaved {len(storm_paths)} images to {outdir}")

    # zip files for transfer to remote training server
    zipdir = traindir / 'zipfiles' / (rescale_method + str(rescale_arg)) / domain
    os.makedirs(zipdir, exist_ok=True)
    zippath = zipdir / (output_format + ".zip")
    print(f"Zipping images to {zippath} ...")
    os.system(f"cd {outdir} && zip -r {zippath} . -i '*.png'")
    print("Done.")


if __name__ == "__main__":
    for method in RESCALE_METHODS:
        for arg in RESCALE_ARGS:
            for domain in DOMAINS:
                for fmt in FORMATS:
                     main(method, arg, domain, fmt)
# %% 
