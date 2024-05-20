"""
Environment: general

Process CMIP6 data daily maximum wind speed at 10m.

References:
-----------
    https://carpentries-lab.github.io/python-aos-lesson/10-large-data/index.html
"""

# %%
import os
import warnings

os.environ["USE_PYGEOS"] = "0"
import glob
import numpy as np
import xarray as xr
import cf_xarray as cfxr
# import xesmf as xe
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = "/Users/alison/Documents/DPhil/data"
indir = os.path.join(datadir, "era5", "new_data")
outdir = os.path.join(wd, "era5_data")

# load all files and process to geopandas
files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.drop_vars(["v10"])
ds_full = ds.copy()
#%% max pool to reduce dimensions (preserves extremes)
pool = 3
r = ds.rolling(latitude=pool, longitude=pool)
rolled = r.construct(latitude="lat", longitude="lon", stride=pool)
ds = rolled.max(dim=['lat', 'lon'], keep_attrs=True)
ds['msl'] = rolled['msl'].min(dim=['lat', 'lon'], keep_attrs=True)
assert ds['msl'].min().values > 1000, "Check sea-level pressure values"

# ref code to stack and unstack 
ds_stacked = ds.stack(grid=["latitude", "longitude"], create_index=True)
df = ds_stacked.to_dataframe()

# save to netcdf
year0 = ds['time'].dt.year.values.min()
yearn = ds['time'].dt.year.values.max()
ds.to_netcdf(os.path.join(outdir, f"data_{year0}_{yearn}.nc"))
# %%
# check looks okay over spatial domain
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ds.isel(time=0).u10.plot(ax=axs[0])
ds.isel(time=0).msl.plot(ax=axs[1])
axs[0].set_title("10m wind")
axs[1].set_title("sea-level pressure")

# %% turn to dataframe --maybe change R code to be ncdf compatible later
df['i'] = df['latitude'].rank(method='dense').astype(int)
df['j'] = df['longitude'].rank(method='dense').astype(int)
df['grid'] = df[['i', 'j']].apply(tuple, axis=1).rank(method='dense').astype(int)
df = df.drop(columns=['latitude', 'longitude']).reset_index()
# %%
df.reset_index().to_parquet(os.path.join(outdir, f"data_{year0}_{yearn}.parquet"))
# %%