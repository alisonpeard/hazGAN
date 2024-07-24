"""
Environment: hazGAN

Process ERA5 data daily maximum wind speed at 10m and minimum MSLP.

Input:
------
    - resampled ERA5 data (netcdf) of form <datadir>/era5/bay_of_bengal__monthly/resampled/*bangladesh*.nc
        (resampled using resample-era5.py script)

Output:
-------
    - netcdf file of processed data (max wind speed, min MSLP) in <target_dir>/training/res_<>x<>/data_{year0}_{yearn}.nc
    - parquet file of processed data in target_dir/training/res_<>x<>/data_{year0}_{yearn}.parquet
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
res = (22, 18)
HOME = '/soge-home/projects/mistral/alison/hazGAN'

source_dir = os.path.join(HOME, 'bay_of_bengal__daily', 'resampled', f'res_{res[1]}x{res[0]}')
target_dir = os.path.join(HOME, "training", f'res_{res[1]}x{res[0]}')

files = glob.glob(os.path.join(source_dir, f"*bangladesh*.nc"))

ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.drop_vars(["v10"])
#%% resample to daily max
ds = ds.dropna(dim='time', how='all') # some dates are missing because API still working
ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})
grid = np.arange(0, res[1] * res[0], 1).reshape(res[1], res[0])
grid = xr.DataArray(
    grid, dims=["latitude", "longitude"], coords={"latitude": ds.latitude[::-1], "longitude": ds.longitude}
)
ds['grid'] = grid
#%% ----Save to netcdf-----
year0 = ds['time'].dt.year.values.min()
yearn = ds['time'].dt.year.values.max()
ds.to_netcdf(os.path.join(target_dir, f"data_{year0}_{yearn}.nc"))
# %% ---- Visualise -----
if False:
    t0 = 0
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ds.isel(time=t0).u10.plot(ax=axs[0])
    ds.isel(time=t0).msl.plot(ax=axs[1])
# %%