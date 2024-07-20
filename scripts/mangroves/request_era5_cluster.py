"""
Augment Yu's data
    - Add Athanisou's slope data
    - Add ERA5 pressure and wind at landfall time at each patch (no more Holland)
To do:
    - Extract lifetime maximum wind instead of just the wind at the time of landfall
"""
#%%
import os
import pandas as pd
import geopandas as gpd
bob_crs = 24346

# %% new way to organise files
indir = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'
outdir = indir
infile1 = os.path.join(indir, 'data_with_ibtracs.gpkg')
infile3 = os.path.join(outdir, 'data_with_ibtracs.csv')
outfile2 = os.path.join(outdir, 'data_with_era5.gpkg')

# %% ----- Get ERA5 reanalysis data ------

import numpy as np
import glob
import xarray as xr

vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_precipitation']
datapath = "/Users/alison/Documents/DPhil/data/era5/global/" # will need to modify later
datapath = '/Users/alison/Documents/DPhil/data/era5/bay_of_bengal__monthly.nosync/original/'
files = glob.glob(f"{datapath}/*.nc")
data = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4") # lazy load data
# %%
# load files for all var types
files = []
for var in vars:
    varpath = os.path.join(datapath, vars, 'nc')
    files += glob.glob(f"{varpath}/*.nc")

data = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4") # lazy load data

winds, pressures, precips = [], [], []
for i, row in gdf.iterrows():
    # get the storm data
    lon = row.lon
    lat = row.lat
    times = row.times
    storm_reanalysis = data.sel(time=times, longitude=lon, latitude=lat, method='nearest')

    # aggregate along time dimension
    u10 = storm_reanalysis['u10'].max(dim=['time', 'latitude', 'longitude'], keep_attrs=True).values
    v10 = storm_reanalysis['v10'].max(dim=['time', 'latitude', 'longitude'], keep_attrs=True).values
    msl = storm_reanalysis['msl'].min(dim=['time', 'latitude', 'longitude'], keep_attrs=True).values
    precip = storm_reanalysis['tp'].sum(dim=['time', 'latitude', 'longitude'], keep_attrs=True).values

    winds.append(np.sqrt(u10**2 + v10**2))
    pressures.append(msl)
    precips.append(precip)

gdf['era5_wind'] = winds
gdf['era5_pressure'] = pressures
gdf['era5_precip'] = precips
# %%

gdf.to_file(outfile2)