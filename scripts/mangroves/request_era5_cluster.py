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
# %%
path = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2/result/model/input_fixedCNTRY_rmOutlier.csv'
df = pd.read_csv(path)
[*df.columns]
df.landingYear.describe() # 2001 - 2017 => can use soge-data
# %% -----Add Athanisou's slope data------
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
slopepath = "/Users/alison/Documents/DPhil/data/athanisou-slopes-2019/nearshore_slopes.csv"
slope_df = pd.read_csv(slopepath)
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
gdf = gpd.sjoin_nearest(gdf.to_crs(bob_crs), slope_gdf[['slope', 'geometry']].to_crs(bob_crs), how='left', distance_col='distance').to_crs(4326)
gdf.sort_values(by='slope', ascending=True).plot('slope', cmap='YlOrRd', legend=True)

# %% -----Import IBTrACs data------
ibtracs = pd.read_csv('/Users/alison/Documents/DPhil/data/ibtracs/ibtracs_since1980.csv')
ibtracs = ibtracs.groupby(['SID']).agg({'NAME': 'first', 'ISO_TIME': list}).reset_index()
ibtracs = ibtracs[['NAME', 'ISO_TIME']].rename(columns={'ISO_TIME': 'times'})
gdf['stormName'] = gdf['stormName'].str.upper() 
gdf = pd.merge(gdf, ibtracs, left_on='stormName', right_on='NAME', how='inner')
gdf = gdf.groupby(['center_centerLat', 'center_centerLon', 'stormName']).agg({'NAME': 'first', 'times': 'first'}).reset_index()
gdf = gdf[['stormName', 'center_centerLat', 'center_centerLon', 'times']]
gdf.columns = ['storm', 'lat', 'lon', 'times']
# %% TODO: Create and ERA5 API request for pressure and wind
# home = os.path.expandvars("$HOME")
# datapath = f'/soge-home/data/analysis/era5/0.28125x0.28125/hourly/' # cluster
import glob
import xarray as xr
datapath = "/Users/alison/Documents/DPhil/data/era5/bay_of_bengal__monthly.nosync/original" # will need to modify later
files = glob.glob(f"{datapath}/*.nc")
data = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4") # lazy load data

# %%
i, row = next(iter(gdf.iterrows()))
# %%
times = row.times[0]
storm_reanalysis = data.sel(longitude=row.lon, latitude=row.lat, method='nearest')

# extract max wind, min pressure, total rainfall
# %%
vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'total_precipitation']

def get_era5_data(i, year, month, day, hour, lat, lon, eps=0.1):
    if not os.path.exists(f'{home}/mistral/alison/era5/mangrove_era5/mangrove_damage_locations_{i}.nc'):
        

        # for var in vars
            # load netcdf 
            # subset by lat, lon, and within time range
            # extract max for wind, precipitation, min for pressure within time range
            # create a single dataarray

        # combine all the dataarrays into a single dataset
        # save the dataset

# do this for each storm 


export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/hazGAN-GPU/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

# %% -----Grab max wind lifetime and min lifetime pressure-----
import glob
import xarray as xr
from tqdm import tqdm

filedir = '/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations'
# add progress bar
for i, row in (pbar := tqdm(gdf[['NAME', 'times', 'center_centerLat', 'center_centerLon']].iterrows(), total=gdf.shape[0])):
    name = row['NAME']
    if not os.path.exists(f'{filedir}/mangrove_damage_locations_{name}_{i}.nc'):
        pbar.set_description(f'Requesting {name} {i}')
        lat = row['center_centerLat']
        lon = row['center_centerLon']
        times = row['times']
        j = 0
        for time in times:
            time = pd.to_datetime(time)
            year = time.year
            month = time.month
            day = time.day
            hour = str(time.hour).zfill(2)
            hour = f"{hour}:00"
            get_era5_data(f"{i}_{name}_{j}", year, month, day, hour, lat, lon)
            j += 1

        files = glob.glob(f"{filedir}/mangrove_damage_locations_{i}_{name}_*.nc")
        ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
        max_wind = (ds['u10']**2 + ds['v10']**2)**0.5
        max_wind = max_wind.max(dim='time', keep_attrs=True).rename('wind')
        min_press = ds['msl'].min(dim='time', keep_attrs=True).rename('mslp')
        ds = xr.merge([max_wind, min_press])
        ds.to_netcdf(f'{filedir}/mangrove_damage_locations_{name}_{i}.nc')

        for file in files:
            os.remove(file)
    else:
        pbar.set_description(f'File for {name} {i} already exists')

