"""
Resample ERA5 (or other gridded) data to a lower resolution using GDAL

Usage:
    - python resample_era5.py -y 1984
    - python resample_era5.py -y 1984 -r 64 64
    - python resample_era5.py -y 1984 --redo
    - use in SLURM with resample.sh

Input files:
    - ERA5 data in netCDF format
    - Files should have variables 'u10', 'v10', 'msl', 'tp'
    - Files should be in a directory named 'original' in the ERA5DIR
    - Files should be named in the format '*YYYY.nc'

Output files:
    - Resampled data in netCDF format
    - Files will be saved in a directory named 'resampled' in the ERA5DIR
    - Files will be saved in a subdirectory named 'resampled/64x64' (or other resolution)
"""
# %%
import os
from environs import Env
import numpy as np
import xarray as xr
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    print('Beginning resampling script!')
    # parse args for resolution with argparse
    parser = argparse.ArgumentParser(description='Resample ERA5 data')
    parser.add_argument('-y', '--year', type=int, help='Year to resample')
    parser.add_argument('-r', '--resolution', type=int, nargs=2, default=[64, 64], help='Resolution of output data (lon, lat)')
    parser.add_argument('--redo', action='store_true', default=False, help='Redo resampling')
    args = parser.parse_args()
    year = args.year
    res = args.resolution
    redo = args.redo

    # set up directories
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("ERA5DIR")
    source_dir = os.path.join(datadir,'input')
    target_dir = os.path.join(datadir,'resampled')
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(source_dir)

    files = os.listdir(source_dir)
    vars = ['u10', 'v10', 'msl', 'tp']
    methods = ['max', 'max', 'min', 'sum']

    files = [f for f in files if str(year) in f]
    print(f"Found {len(files)} files for year {year}")
    print(f"Resampling to {res[1]}x{res[0]}")

    for file_long in (pbar := tqdm(files)):
        file = file_long.split('.')[0]

        if not redo and os.path.exists(os.path.join(target_dir, f"{file}.nc")):
            continue

        if file_long[0] == '.' or file_long[-3:] != '.nc':
            pbar.update_description(f'Skipping {file_long}')
            continue

        pbar.set_description(f'Resampling {file_long}')
        ds_orig = xr.open_dataset(f"{file}.nc") #os.path.join(source_dir, f"{file}.nc")

        # pretty print ds_orig summary
        print(ds_orig)


        times = ds_orig.time.values
        resampled_datasets = []
        for var, method in zip(vars, methods):
            command = f'gdalwarp -t_srs EPSG:4326 -ts {res[0]} {res[1]} -r {method} -overwrite -of netCDF "NETCDF:{file}.nc:{var}" ../resampled/{res[1]}x{res[0]}/{file}_{var}.nc'
            print("Submitting command:", command)
            os.system(command) # use GDAL to resample

            ds_var = xr.open_dataset(os.path.join(target_dir, f"{file}_{var}.nc"))
            bands = [var for var in ds_var.data_vars if 'Band' in var]
            ds_var = ds_var[bands].to_array('time', name=var).to_dataset().assign_coords(time=times)

            opt_func = getattr(np.ndarray, method)
            # assert opt_func(ds_orig[var].values) == opt_func(ds_var[var].values), f"Check {var} not smoothened"
            resampled_datasets.append(ds_var)
            
        ds = xr.merge(resampled_datasets)
        ds.to_netcdf(os.path.join(target_dir, f"{file}.nc"))
        print(f"Saved {file}.nc")

        # cleanup temporary files
        for var in vars:
            os.remove(os.path.join(target_dir, f"{file}_{var}.nc"))
#%%

