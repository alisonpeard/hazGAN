# %%
import os
import numpy as np
import xarray as xr

redo = False
source_dir = "/Users/alison/Documents/DPhil/data/era5/new_data.nosync"
target_dir = "/Users/alison/Documents/DPhil/data/era5/new_data.nosync/resampled"

os.chdir(source_dir)
files = os.listdir(source_dir)
vars = ['u10', 'v10', 'msl']
methods = ['max', 'max', 'min']
print(files)
# %%
for file_long in files:
    file = file_long.split('.')[0]

    if not redo and os.path.exists(os.path.join(target_dir, f"{file}.nc")):
        continue

    if file_long[0] == '.' or file_long[-3:] != '.nc':
        print('Skipping', file_long)
        continue

    ds_orig = xr.open_dataset(os.path.join(source_dir, f"{file}.nc"))
    times = ds_orig.time.values
    resampled_datasets = []
    for var, method in zip(vars, methods):
        command = f'gdalwarp -t_srs EPSG:4326 -ts 22 18 -r {method} -overwrite -of netCDF NETCDF:\\"{file}.nc\\":{var} resampled/{file}_{var}.nc'
        os.system(command) # use GDAL to resample
        ds_var = xr.open_dataset(os.path.join(target_dir, f"{file}_{var}.nc"))
        bands = [var for var in ds_var.data_vars if 'Band' in var]
        ds_var = ds_var[bands].to_array('time', name=var).to_dataset().assign_coords(time=times)

        opt_func = getattr(np.ndarray, method)
        assert opt_func(ds_orig[var].values) == opt_func(ds_var[var].values), f"Check {var} not smoothened"
        resampled_datasets.append(ds_var)
    ds = xr.merge(resampled_datasets)
    ds.to_netcdf(os.path.join(target_dir, f"{file}.nc"))

    # cleanup temporary files
    for var in vars:
        os.remove(os.path.join(target_dir, f"{file}_{var}.nc"))

#%%