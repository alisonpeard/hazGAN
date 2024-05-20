# %%
import os
import numpy as np
import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# %%
wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = "/Users/alison/Documents/DPhil/data"
indir = os.path.join(datadir, "era5", "new_data")
outdir = os.path.join(wd, "era5_data")

# load all files and process to geopandas
files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


dates = pd.read_csv('/Users/alison/Documents/DPhil/multivariate/era5_data/ibtracs_dates.csv')
dates = pd.to_datetime(dates['time'])

ds_ts = ds.sel(time=ds.time.isin([dates]))
# %% contour plot https://doi.org/10.1038/s41612-024-00638-w 
fig = plt.figure(figsize=(10,8))
ax  = plt.axes(projection =ccrs.PlateCarree()) 
ax.coastlines(resolution='50m',color='k', linewidth=.5) 
t0 = 1225 # 2022-12-09 looks good
# ds_ts.where(ds_ts.msl == ds.msl.min())
artist = xr.plot.contour(ds_ts.isel(time=t0).msl, colors='lightgrey', levels=20, linewidths=.5, ax=ax, zorder=2)
Pa = artist.levels[17]
xr.plot.contourf(ds_ts.where(ds_ts.msl <= Pa).isel(time=t0).u10, fill=True, cmap='plasma_r', levels=10, ax=ax, zorder=1)
print(Pa)
# %% example

# %%
