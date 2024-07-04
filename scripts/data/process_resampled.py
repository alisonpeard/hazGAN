"""
Environment: hazGAN

Process ERA5 data daily maximum wind speed at 10m and minimum MSLP.

Input:
------
    - resampled ERA5 data (netcdf) of form <datadir>/era5/new_data/resampled/*bangladesh*.nc
        (resampled using resample-era5.py script)

Output:
-------
    - netcdf file of processed data (max wind speed, min MSLP) in <outdir>/era5_data/data_{year0}_{yearn}.nc
    - parquet file of processed data in outdir/era5_data/data_{year0}_{yearn}.parquet
"""

# %%
import os
import warnings

os.environ["USE_PYGEOS"] = "0"
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %% -----Load data-----
wd = "/Users/alison/Documents/DPhil/multivariate"
res = (28, 28)
indir = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'data', 'era5', 'new_data.nosync', 'resampled', f'res_{res[1]}x{res[0]}')
outdir = os.path.join(wd, "era5_data.nosync", f'res_{res[1]}x{res[0]}')

files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))

ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.drop_vars(["v10"])
#%% resample to daily max
ds_resampled = ds.resample(time="1D").max()
ds_resampled['msl'] = ds.msl.resample(time="1D").min()
ds = ds_resampled
ds = ds.dropna(dim='time', how='all') # some dates are missing because API still working
ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})
grid = np.arange(0, res[1] * res[0], 1).reshape(res[1], res[0])
grid = xr.DataArray(
    grid, dims=["latitude", "longitude"], coords={"latitude": ds.latitude[::-1], "longitude": ds.longitude}
)
ds['grid'] = grid
ds.grid.plot()
#%% ----Save to netcdf-----
year0 = ds['time'].dt.year.values.min()
yearn = ds['time'].dt.year.values.max()
ds.to_netcdf(os.path.join(outdir, f"data_{year0}_{yearn}.nc"))
# %% ---- Visualise -----
t0 = 16537
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ds.isel(time=t0).u10.plot(ax=axs[0])
ds.isel(time=t0).msl.plot(ax=axs[1])
# %%