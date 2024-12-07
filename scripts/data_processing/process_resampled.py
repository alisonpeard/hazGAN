"""
Environment: hazGAN

Need to go back to this once I have found the source of the NaNs.

Process ERA5 data daily variables with the following aggregations:
    - sqrt(u10^2 + v10^2): maximum
    - mslp: minimum
    - tp: sum

Remove wind bombs using a similarity metric based on Frobenius norm.

Input:
------
    - resampled ERA5 data (netcdf) of form
        <datadir>/era5/bay_of_bengal__monthly/resampled/<resolution>*bangladesh*.nc
        (resampled using resample_era5.py script)

Output:
-------
    - netcdf file of processed data (max wind speed, min MSLP) in <target_dir>/training/res_<>x<>/data_{year0}_{yearn}.nc
    - parquet file of processed data in target_dir/training/res_<>x<>/data_{year0}_{yearn}.parquet
"""
# %%
import os
os.environ["USE_PYGEOS"] = "0"
from glob import glob
from time import time
import numpy as np
import pandas as pd
import xarray as xr
from environs import Env
from collections import Counter
import matplotlib.pyplot as plt

VISUALS = True
VIEW = 5
THRESHOLD = 0.98 # human-implemented bisection algorithm


def rescale(x:np.ndarray) -> np.ndarray:
    return (x - x.min() / (x.max() - x.min()))


def rescale_vector(x:np.ndarray) -> np.ndarray:
    return (x - x.min(axis=(1, 2), keepdims=True)) / (x.max(axis=(1, 2), keepdims=True) - x.min(axis=(1, 2), keepdims=True))


def frobenius(test:np.ndarray, template:np.ndarray) -> np.ndarray:
    sum_ = np.sum(template * test, axis=(1, 2))
    norms = np.linalg.norm(template) * np.linalg.norm(test, axis=(1, 2))
    similarities = sum_ / norms
    return similarities


def get_similarities(ds:xr.Dataset, template:np.ndarray) -> np.ndarray:

    template = rescale(template)
    tensor = ds['u10'].data
    tensor = rescale_vector(tensor)
    similarities = frobenius(tensor, template)
    
    return similarities # np.array(similarities)


if __name__ == "__main__":
    # setup environment
    env = Env()
    env.read_env(recurse=True)
    source_dir = env.str("ERA5DIR")
    target_dir = env.str("TRAINDIR")
    files = glob(os.path.join(source_dir, "resampled", "18x22", f"*bangladesh*.nc"))
    start = time()

    # load data
    ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
    ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    ds = ds.drop_vars(["v10"])
    print("Number of days of data: {}".format(ds.dims['time']))
    print("Number of years found: {}".format(len(np.unique(ds['time.year'].data))))

    # check for missing dates
    start_date = ds['time'].dt.date.min().data.item()
    end_date = ds['time'].dt.date.max().data.item()
    all_dates = pd.date_range(start_date, end_date, freq='D').date
    ds_dates = ds['time'].dt.date.data
    missing_dates = set(all_dates) - set(ds_dates)
    missing_year_counts = Counter([d.year for d in sorted(missing_dates)])
    print("\nMissing years:\n--------------")
    print('\n'.join([f"{k}: {v} days" for k, v in missing_year_counts.items()]))

    # resample to daily maxima
    ds = ds.dropna(dim='time', how='all')
    h, w = ds.dims['lat'], ds.dims['lon']
    grid = np.arange(0, h * w, 1).reshape(h, w)
    grid = xr.DataArray(
        grid, dims=["lat", "lon"], coords={"lat": ds.lat[::-1], "lon": ds.lon}
    )
    ds['grid'] = grid

    # ----Remove "wind bombs"-----
    ds['maxwind'] = ds['u10'].max(dim=['lat', 'lon'])
    ds = ds.sortby('maxwind', ascending=False)

    if VISUALS:
        fig, axs = plt.subplots(VIEW, VIEW)
        for i, ax in enumerate(axs.ravel()):
            ds.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
        fig.suptitle(f'{VIEW*VIEW} most extreme wind fields')

    print("Processing data for wind bombs...")
    template = ds.isel(time=0).u10.data
    similarities = get_similarities(ds, template)
    similarities = similarities.compute()

    if VISUALS:
        order = np.argsort(similarities)[::-1]
        ds_ordered = ds.isel(time=order).copy()

        fig, axs = plt.subplots(VIEW, VIEW)
        for i, ax in enumerate(axs.ravel()):
            ds_ordered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
        fig.suptitle(f'{VIEW*VIEW} winds most similar to wind bomb template')

    # remove wind bombs
    nbombs = sum(similarities > THRESHOLD)
    print(f'{nbombs} ERA5 "wind bombs" detected in dataset for threshold {THRESHOLD}.')
    mask = (similarities <= THRESHOLD)
    ds_filtered = ds.isel(time=mask)
    ds_filtered = ds_filtered.sortby('maxwind', ascending=False)

    if VISUALS:
        fig, axs = plt.subplots(VIEW, VIEW)
        for i, ax in enumerate(axs.ravel()):
            ds_filtered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            fig.suptitle(f'{VIEW*VIEW} most extreme winds after filtering')

    # ----save to netcdf-----
    year0 = ds_filtered['time'].dt.year.values.min()
    yearn = ds_filtered['time'].dt.year.values.max()
    ds_filtered.to_netcdf(os.path.join(target_dir, f"data_{year0}_{yearn}.nc"))

    end = time()
    print(f"Processing time: {end - start:.2f} seconds")

    # final visualisation
    if VISUALS:
        t0 = 0
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        ds.isel(time=t0).u10.plot(ax=axs[0])
        ds.isel(time=t0).msl.plot(ax=axs[1])
        ds.isel(time=t0).tp.plot(ax=axs[2])
# %%