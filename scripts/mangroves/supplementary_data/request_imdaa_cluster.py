
#%%
import os
import glob
import xarray as xr
import numpy as np
import dask

vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_precipitation']
vars = ['ugrd', 'vgrd', 'gust', 'prmsl', 'apcp']
# %% new way to organise files
HOME = '/Volumes'
# HOME = '/soge-home/projects'

indir = os.path.join(HOME, 'mistral/incoming_data/ncmrwf-reanalysis/')
outdir = os.path.join(HOME,'mistral/alison/hazGAN/mangrove')
infile1 = os.path.join(outdir, 'data_with_era5_mini.csv') # TODO: change once working
outfile2 = os.path.join(outdir, 'data_with_imdaa.csv')

# %% ----- Get ERA5 reanalysis data ------
import pandas as pd
from ast import literal_eval 
df = pd.read_csv(infile1, index_col=0)

df['times'] = df['times'].apply(literal_eval) 
df['years'] = df['times'].apply(lambda x: list(set(pd.to_datetime(y, format='%Y-%m-%d %H:%M:%S').year for y in x)))
df['start_year'] = df['years'].apply(min)
years = list(set(item for sublist in df['years'] for item in sublist))
storms = df['storm'].unique()   
# %% load files for all var types
files = []
for var in vars:
    varpath = os.path.join(indir, var)
    files += glob.glob(f"{varpath}/*.nc")

files_all = files.copy()
files = [file for file in files if any(str(year) in file for year in years)]
# data = xr.open_mfdataset(files, chunks={"time": "500MB", 'latitude': 50, 'longitude': 50}, engine="netcdf4") # lazy load data

# %%
from tqdm import tqdm

# extract data by year
winds, gusts, pressures, precips = [], [], [], []
for year in (pbar := tqdm(years)):
    df_year = df[df['start_year'] == year]
    year_list = [str(year), str(year+1)]
    files_sub = [file for file in files if any(year in file for year in year_list)]
    pbar.set_description(f"Loading {year}")
    data = xr.open_mfdataset(files_sub, engine='netcdf4', chunks={"time": "500MB"}) # lazy load data
    # might be smart to subset by storm aswell
    storms = list(df_year['storm'].unique())
    for storm in (pbar2 := tqdm(storms, leave=False)):
        pbar2.set_description(f" -- Processing Storm {storm.capitalize()}")
        df_storm = df_year[df_year['storm'] == storm].reset_index(drop=True)
        times = df_storm['times'].values[0]
        storm = data.sel(time=times, method='nearest')

        # Vectorized selection
        lons = df_storm['lon'].values
        lats = df_storm['lat'].values
        storm_patches = storm.sel(longitude=xr.DataArray(lons), latitude=xr.DataArray(lats), method='nearest')

        # Vectorized computation
        u10_max = storm_patches['UGRD_10m'].max(dim='time').values
        v10_max = storm_patches['VGRD_10m'].max(dim='time').values
        gust_max = storm_patches['GUST_10m'].max(dim='time').values
        msl_min = storm_patches['PRMSL_msl'].min(dim='time').values
        precip_sum = storm_patches['APCP_sfc'].sum(dim='time').values
        
        # Append results
        winds.extend(np.sqrt(u10_max**2 + v10_max**2))
        gusts.extend(gust_max)
        pressures.extend(msl_min)
        precips.extend(precip_sum)

# Compute results (if using dask)
winds, gusts, pressures, precips = dask.compute(winds, gusts, pressures, precips)

# %%
df['imdaa_wind'] = winds
df['imdaa_gust'] = gusts
df['imdaa_pressure'] = pressures
df['imdaa_precip'] = precips
# %%

df.to_csv(outfile2)
# %%