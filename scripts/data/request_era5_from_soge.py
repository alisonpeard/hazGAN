
#%%
import os
import sys
from glob import glob
import xarray as xr
import numpy as np
import dask

i = int(sys.argv[1]) # load the index from the command line

xmin, xmax =  80., 95.
ymin, ymax = 10., 25.

years = np.arange(1940, 2023)
year = years[i]

variables = {
    'u10': 'max',
    'v10': 'max',
    'msl': 'min',
    'tp': 'sum'
}
var_long = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'msl': 'mean_sea_level_pressure',
    'tp': 'total_precipitation'
}

# HOME = '/Volumes'       # if connecting from local (dev only)
HOME = '/soge-home/'  # if connecting from cluster
indir = os.path.join(HOME, 'data/analysis/era5/0.28125x0.28125/hourly/')
outdir = os.path.join(HOME,'projects/mistral/alison/hazGAN/bay_of_bengal__daily/original')
# %% 
files = []
for var_name in var_long.values():
    var_files = glob(os.path.join(indir, var_name, 'nc', '*'))
    files += var_files[:1]
files_year = [f for f in files if str(year) in f]

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    data = xr.open_mfdataset(files_year, engine='netcdf4', chunks={"time": "500MB", 'longitude': '500MB', 'latitude': '500MB'})
    data = data.sel(longitude=slice(xmin, xmax), latitude=slice(ymax, ymin))

resampled = {}
for var, func in variables.items():
    resampled[var] = getattr(data[var].resample(time='1D'), func)()

data_resampled = xr.Dataset(resampled)
chunk_size = {'time': '500MB'}
data_resampled = data_resampled.chunk(chunk_size)
output_file = os.path.join(outdir, f'bangladesh_{year}.nc')

data_resampled.to_netcdf(output_file, compute=False)
data.close()
data_resampled.close()
# %%
