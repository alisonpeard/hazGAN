"""
Environment: general

Process ERA5 data daily maximum wind speed at 10m and minimum MSLP.

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

wind_max = ds_full.u10.values.max()
pressure_min = ds_full.msl.values.min()
# %%
def output_size(h_in, k, s, p=0):
    h_out = np.floor((h_in + 2 * p - k) / s) + 1
    return int(h_out)

# %%
def find_kernel_stride_pairs(H_in, H_out):
    pairs = []
    for stride in range(1, H_in):
        kernel_size_numerator = H_in - (H_out - 1) * stride
        if kernel_size_numerator > 0 and kernel_size_numerator <= H_in:
            for remainder in range(stride):
                kernel_size = kernel_size_numerator + remainder
                if kernel_size <= H_in:
                    pairs.append((kernel_size, stride))
    return pairs

# Example usage
H_in = 10
H_out = 5
possible_pairs = find_kernel_stride_pairs(H_in, H_out)
print(possible_pairs)
# output_size(61, 4, 3, 1)
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
assert ds.u10.values.max() == wind_max, "Check max wind speed not smoothened"
assert ds.msl.values.min() == pressure_min, "Check min pressure not smoothened"
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
# %% saffir-simpson scale
# https://www.nhc.noaa.gov/aboutsshws.php


ss_u10 = np.array([16, 29, 38, 44, 52, 63])
ss_msl = np.array([925, 945, 960, 975, 990, 1005]) * 100

msl = ds_full.msl.values
u10 = ds_full.u10.values

# %%
plt.style.use('seaborn-v0_8')
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(msl.ravel(), bins=50, color='lightgrey', edgecolor='k') 
axs[1].hist(u10.ravel(), bins=50, color='lightgrey', edgecolor='k')

for i, (ss_i, msl_i) in enumerate(zip(ss_u10, ss_msl)):
    axs[0].axvline(msl_i, color='c', linestyle='--', label=f"SS {i+1} {msl_i}hPa")
    axs[1].axvline(ss_i, color='c', linestyle='--', label=f"SS {i+1} {ss_i}m/s")
axs[0].set_title("Sea-level pressure")
axs[1].set_title("10m wind speed")
axs[0].legend()
axs[1].legend()
# %%
