# %%
import os
from environs import Env
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

from hazGAN import statistics

WINDTHRESHOLD = [15, -float("inf")][0]
EPS           = 1e-6
RES  = (64, 64)
CMAP = "Spectral_r"

i, j = 3, 1
DOMAIN = ["rescaled", "uniform", "gumbel", "gaussian"][i]
RESCALE_METHOD = ["minmax", "rp"][j]
RESCALE_ARG = [0.9, 10_000][j]

def rescale_array(array, method="minmax09", arg=RESCALE_ARG, domain=DOMAIN):

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


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    traindir = env.str("TRAINDIR")

    print(f"Loading training data from {traindir}")
    ds = xr.open_dataset(os.path.join(traindir, 'data.nc'))
    ds['windmax'] = ds.sel(field='u10').anomaly.max(dim=['lon', 'lat'])
    mask = (ds['windmax'] > WINDTHRESHOLD).values
    idx  = np.where(mask)[0]
    ds   = ds.isel(time=idx)

    print(f"\nFound {ds.time.size} training events with maximum wind exceeding {WINDTHRESHOLD} m/s")

    outdir = os.path.join(traindir, 'images', f"{DOMAIN}_{RESCALE_METHOD}{str(RESCALE_ARG).replace('.', '')}", "png")
    stats_path = os.path.join(outdir, "..", "image_stats.npz")
    os.makedirs(os.path.join(outdir, "png"), exist_ok=True)

    nimgs = ds.time.size

    if DOMAIN == "rescaled":
         u = ds.anomaly.values
    else:
        u = ds.uniform.values
        print(f"Maximum u-value found is {u.max():.6f}")
        print(f"Corresponds to {1/(1-u.max()):,.0f}-year return level assumption")
        print(f"Minimum u-value found is {u.min():.6f}")
        print(f"Corresponds to {1/(1-u.min()):,.0f}-year return level assumption")

        if not ((u.max() < 1.0) and (u.min() >= 0.0)):
            raise ValueError("Percentiles not in [0, 1) range")
    
    u = np.flip(u, axis=1) #! flip latitude

    assert u.shape[1:] == (64, 64, 3), f"Unexpected shape: {u.shape}"

    ppf = getattr(statistics, DOMAIN)
    y = ppf(u)
    y, stats = rescale_array(y, method=RESCALE_METHOD, arg=RESCALE_ARG, domain=DOMAIN)
    np.savez(stats_path, **stats)

    # save images
    for i in range(nimgs):
        y_i = y[i]
        assert np.all((y_i >= 0.) & (y_i < 1.)), \
            f"Array values out of [0,1) range: min {y_i.min()}, max {y_i.max()}"
        y_i = np.uint8(y_i * 255)
        img = Image.fromarray(y_i, 'RGB')
        output_path = os.path.join(outdir, f"storm_{i}.png")
        img.save(output_path)

    storm_paths = sorted(glob.glob(os.path.join(outdir, "storm_*.png")))
    print(f"\nSaved {len(storm_paths)} images to {outdir}")

    zipdir = os.path.join(traindir, 'images', 'zipfiles')
    os.makedirs(zipdir, exist_ok=True)
    zippath = os.path.join(zipdir, f"{DOMAIN}_{RESCALE_METHOD}{str(RESCALE_ARG).replace('.', '')}.zip")
    print(f"Zipping images to {zippath} ...")
    # os.system(f"cd {outdir} && zip -r {zippath} .")
    # zip only .png files
    os.system(f"cd {outdir} && zip -r {zippath} . -i '*.png'")
    print("Done.")


# %% 
