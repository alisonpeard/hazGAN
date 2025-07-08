
"""
Load ERA5 data from hourly netcdf files, resample to daily aggregates, and save to a single netcdf file in the target directory.
"""

import os
import sys
from environs import Env
import dask
import time
from glob import glob
from pprint import pp as prettyprint
import xarray as xr
import numpy as np

XMIN, XMAX = 80., 95.
YMIN, YMAX = 10., 25.

YEAR0 = 1940
YEARN = 2023

VARIABLES = {
    'u10': 'max',
    'v10': 'max',
    'msl': 'min',
    'tp': 'sum'
}

VAR_LONG = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'msl': 'mean_sea_level_pressure',
    'tp': 'total_precipitation'
}

if __name__ == '__main__':
    print('Beginning data acquisiton')
    start = time.time()
    env = Env()
    env.read_env(recurse=True)
    source_dir = '/soge-home/data/analysis/era5/0.28125x0.28125/hourly/'
    target_dir = os.path.join(env.str('ERA5DIR'), 'original')

    i = int(sys.argv[1]) # load the index from the command line
    years = np.arange(YEAR0, YEARN)
    year = years[i]
    
    # load data
    files = []
    for var_name in VAR_LONG.values():
        var_files = glob(os.path.join(source_dir, var_name, 'nc', '*'))
        files += var_files
    files_year = [f for f in files if str(year) in f]
    print(f"Found {len(files_year)} files for {year}...")

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        data = xr.open_mfdataset(files_year, engine='netcdf4', chunks={"time": "500MB", 'longitude': '500MB', 'latitude': '500MB'})
        data = data.sel(longitude=slice(XMIN, XMAX), latitude=slice(YMAX, YMIN))
    print("Data loaded.")

    resampled = {}
    for var, func in VARIABLES.items():
        resampled[var] = getattr(data[var].resample(time='1D'), func)()
    data_resampled = xr.Dataset(resampled)
    print('Data resampled to daily aggregates (min, max, sum).')

    chunk_size = {'time': '50MB'}
    data_resampled = data_resampled.chunk(chunk_size)
    output_file = os.path.join(target_dir, "input", f'{year}.nc')
    print(f"Saving data to {output_file} ...")
    data_resampled.to_netcdf(output_file, engine='netcdf4')
    print(f"Saved. File size: {os.path.getsize(output_file) / 1e6:.2f} MB.")

    data.close()
    data_resampled.close()

    # # load and test time encoding
    # data = xr.open_dataset(output_file, decode_times=False, engine='netcdf4')
    # times = data.isel(time=slice(0, 4)).time.data
    # print(f"Time encoding: {times[0]}, {times[1]}, ...")
    # print(f"Encoding metadata:")
    # prettyprint(data.time.attrs)

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds.")
# %%
