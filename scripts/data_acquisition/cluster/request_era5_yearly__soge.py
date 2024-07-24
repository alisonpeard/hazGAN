
#%%
import os
import sys
from glob import glob
import xarray as xr
import numpy as np
import dask

print('Hi!')
i = int(sys.argv[1]) # load the index from the command line
# i = 0

xmin, xmax =  80., 95.
ymin, ymax = 10., 25.

years = np.arange(1940, 1950) # (1940, 2023)
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
source_dir = os.path.join(HOME, 'data/analysis/era5/0.28125x0.28125/hourly/')
target_dir = os.path.join(HOME,'projects/mistral/alison/hazGAN/bay_of_bengal__daily/original')
# %% 
files = []
for var_name in var_long.values():
    var_files = glob(os.path.join(source_dir, var_name, 'nc', '*'))
    files += var_files
files_year = [f for f in files if str(year) in f]
print(f"Found {len(files_year)} files for year {year}")

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    data = xr.open_mfdataset(files_year, engine='netcdf4', chunks={"time": "500MB", 'longitude': '500MB', 'latitude': '500MB'})
    data = data.sel(longitude=slice(xmin, xmax), latitude=slice(ymax, ymin))
print("Data loaded")

resampled = {}
for var, func in variables.items():
    resampled[var] = getattr(data[var].resample(time='1D'), func)()
data_resampled = xr.Dataset(resampled)
print('Data resampled to daily aggregates (min, max, sum)')

chunk_size = {'time': '500MB'}
data_resampled = data_resampled.chunk(chunk_size)
output_file = os.path.join(target_dir, f'bangladesh_{year}.nc')
data_resampled.to_netcdf(output_file)
print(f"Data saved to {output_file}")

data.close()
data_resampled.close()
print('Bye!')
# %%
