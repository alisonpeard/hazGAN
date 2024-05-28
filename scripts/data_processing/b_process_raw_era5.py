"""
Environment: hazGAN

Process ERA5 data daily maximum wind speed at 10m and minimum MSLP.

Input:
------
    - raw ERA5 data (netcdf) of form <datadir>/era5/new_data/*bangladesh*.nc

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
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# %% DELETE ME
x = xr.open_dataset('/Users/alison/Documents/DPhil/data/era5/new_data/bangladesh_1950_01.nc')
x
# %%
#  read direct from linux server
wd = "/Users/alison/Documents/DPhil/multivariate"
indir = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'data', 'era5', 'new_data')
outdir = os.path.join(wd, "era5_data")

# load all files and process to geopandas
files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.drop_vars(["v10"])
ds_full = ds.copy()

# grab max/min values for later checks
wind_max = ds_full.u10.values.max()
pressure_min = ds_full.msl.values.min()
#%% resample to daily max
ds_resampled = ds.resample(time="1D").max()
ds_resampled['msl'] = ds.msl.resample(time="1D").min()
ds = ds_resampled
ds = ds.dropna(dim='time', how='all') # some dates are missing because API takes forever
assert ds.u10.values.max() == wind_max, "Check max wind speed not smoothened"
assert ds.msl.values.min() == pressure_min, "Check min pressure not smoothened"
#%% max pool to reduce dimensions (preserves extremes)
pool = 3
r = ds.rolling(latitude=pool, longitude=pool)
rolled = r.construct(latitude="lat", longitude="lon", stride=pool)
ds = rolled.max(dim=['lat', 'lon'], keep_attrs=True)
ds['msl'] = rolled['msl'].min(dim=['lat', 'lon'], keep_attrs=True)
assert ds['msl'].min().values > 1000, "Check sea-level pressure values"

# %% when ready replace ds with ds_resampled
assert ds.u10.values.max() == wind_max, "Check max wind speed not smoothened"
assert ds.msl.values.min() == pressure_min, "Check min pressure not smoothened"
# %%
grid = np.arange(0, 21 * 21, 1).reshape(21, 21)
grid = xr.DataArray(
    grid, dims=["latitude", "longitude"], coords={"latitude": ds.latitude[::-1], "longitude": ds.longitude}
)
ds['grid'] = grid
# %%
ds.grid.plot()
#%%
# save to netcdf
year0 = ds['time'].dt.year.values.min()
yearn = ds['time'].dt.year.values.max()
ds.to_netcdf(os.path.join(outdir, f"data_{year0}_{yearn}.nc"))
ds # check out dimensions etc
# %% check looks okay over spatial domain
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ds.isel(time=0).u10.plot(ax=axs[0])
ds.isel(time=0).msl.plot(ax=axs[1])
axs[0].set_title("10m wind")
axs[1].set_title("sea-level pressure")
# %%