"""
Process data resampled using resample.py script and remove "wind bombs" from the dataset.

Process  daily variables with the following aggregations:
    - sqrt(u10^2 + v10^2): maximum
    - mslp: minimum
    - tp: sum

Remove wind bombs using a similarity metric based on Frobenius norm.

Input files:
    - resampled ERA5 data (netcdf) of form
        <ERA5DIR>/resampled/<resolution>/*.nc

Output files:
    - netcdf file of processed data (max wind speed, min MSLP) in <TRAINDIR>/<resolution>/data_{year0}_{yearn}.nc
    - numpy file of wind bomb template in <TRAINDIR>/<resolution>/windbomb.npy

"""
# %%
import os
from glob import glob
from time import time
import numpy as np
import pandas as pd
import xarray as xr
from environs import Env
from collections import Counter
import matplotlib.pyplot as plt

from hazGAN.utils import get_similarities
from hazGAN.utils import res2str
os.environ["USE_PYGEOS"] = "0"

VISUALS = True
VIEW = 4
THRESHOLD = 0.82 # human-implemented bisection algorithm
RES = (64, 64)


if __name__ == "__main__":
    # setup environment
    env = Env()
    env.read_env(recurse=True)
    source_dir = env.str("ERA5DIR")
    target_dir = os.path.join(env.str("TRAINDIR"), res2str(RES))
    files = glob(os.path.join(source_dir, "resampled", res2str(RES), f"*bangladesh*.nc"))
    start = time()

    # load data
    ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
    ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    ds = ds.drop_vars(["v10"])
    print("Number of days of data: {:,}".format(ds.sizes['time']))
    print("Number of years found: {}".format(len(np.unique(ds['time.year'].data))))

    # check for missing dates
    start_date = ds['time'].dt.date.min().data.item()
    end_date = ds['time'].dt.date.max().data.item()
    all_dates = pd.date_range(start_date, end_date, freq='D').date
    ds_dates = ds['time'].dt.date.data
    missing_dates = set(all_dates) - set(ds_dates)
    missing_year_counts = Counter([d.year for d in sorted(missing_dates)])

    if len(missing_dates) > 0:
        print("\nMissing years:\n--------------")
        print('\n'.join([f"{k}: {v} days" for k, v in missing_year_counts.items()]))

    # resample to daily maxima
    ds = ds.dropna(dim='time', how='all')
    h, w = ds.sizes['lat'], ds.sizes['lon']
    grid = np.arange(0, h * w, 1).reshape(h, w)
    grid = xr.DataArray(
        grid, dims=["lat", "lon"], coords={"lat": ds.lat[::-1], "lon": ds.lon}
    )
    ds['grid'] = grid

    # %%----Remove "wind bombs"-----
    ds['maxwind'] = ds['u10'].max(dim=['lat', 'lon'])
    ds = ds.sortby('maxwind', ascending=False)

    if VISUALS:
        fig, axs = plt.subplots(VIEW, VIEW, figsize=(20, 20))
        for i, ax in enumerate(axs.ravel()):
            ds.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
        fig.suptitle(f'{VIEW*VIEW} most extreme wind fields', fontsize=24, y=.95)

    print("Processing data for wind bombs...")
    template = ds.isel(time=0).u10.data
    similarities = get_similarities(ds, template)
    similarities = similarities.compute()
    ds['similarities'] = xr.DataArray(similarities, dims='time')

    # 
    if VISUALS:
        ds_ordered = ds.sortby('similarities', ascending=False)

        fig, axs = plt.subplots(VIEW, VIEW, figsize=(20, 20))
        for i, ax in enumerate(axs.ravel()):
            ds_ordered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            date = ds_ordered.isel(time=i)['time.date'].data.item()
            similarity = ds_ordered.isel(time=i)['similarities'].data.item()
            ax.set_title("{:.2%}: {}".format(similarity, date))

        fig.suptitle(
            f'{VIEW*VIEW} winds most similar to wind bomb template', fontsize=24, y=.95
            )

    # remove wind bombs
    nbombs = sum(similarities >= THRESHOLD)
    print(f'{nbombs} ERA5 "wind bombs" detected in dataset for threshold {THRESHOLD}.')
    mask = (similarities <= THRESHOLD)
    ds_filtered = ds.isel(time=mask)
    ds_filtered = ds_filtered.sortby('maxwind', ascending=False)

    if VISUALS:
        fig, axs = plt.subplots(VIEW, VIEW, figsize=(20, 20))
        for i, ax in enumerate(axs.ravel()):
            ds_filtered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
            ax.axis('off')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            fig.suptitle(f'{VIEW*VIEW} most extreme winds after filtering', fontsize=24, y=0.95)

    # ----save to netcdf-----
    year0 = ds_filtered['time'].dt.year.values.min()
    yearn = ds_filtered['time'].dt.year.values.max()
    np.save(os.path.join(target_dir, "windbomb.npy"), template.compute())
    ds_filtered.to_netcdf(os.path.join(target_dir, f"data_{year0}_{yearn}.nc"))

    end = time()
    print(f"Processing time: {end - start:.2f} seconds")

    # final visualisation
    if VISUALS:
        t0 = 0
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        ds_filtered.isel(time=t0).u10.plot.contourf(ax=axs[0], levels=20, cmap="Spectral_r")
        ds_filtered.isel(time=t0).msl.plot.contourf(ax=axs[1], levels=20, cmap="Spectral")
        ds_filtered.isel(time=t0).tp.plot.contourf(ax=axs[2], levels=20, cmap="PuBu")
# %%