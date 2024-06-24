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
path = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2/input_fixedCNTRY_rmOutlier.csv'
df = pd.read_csv(path)
[*df.columns]
# %% -----Add Athanisou's slope data------
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
slopepath = "/Users/alison/Documents/DPhil/data/athanisou-slopes-2019/nearshore_slopes.csv"
slope_df = pd.read_csv(slopepath)
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
gdf = gpd.sjoin_nearest(gdf.to_crs(bob_crs), slope_gdf[['slope', 'geometry']].to_crs(bob_crs), how='left', distance_col='distance').to_crs(4326)
gdf.sort_values(by='slope', ascending=True).plot('slope', cmap='YlOrRd', legend=True)

# %%
# import ibtracs data
ibtracs = pd.read_csv('/Users/alison/Documents/DPhil/data/ibtracs/ibtracs_since1980.csv')
ibtracs = ibtracs[ibtracs['DIST2LAND'] == 0]
ibtracs = ibtracs[['NAME', 'ISO_TIME', 'DIST2LAND', 'LON', 'LAT']]
ibtracs
# %%
gdf['stormName'] = gdf['stormName'].str.upper() 
gdf = pd.merge(gdf, ibtracs, left_on='stormName', right_on='NAME', how='left')
# %%  now groupb
gdf = gdf[gdf['landingLat'] == gdf['LAT']]
gdf = gdf[gdf['landingLon'] == gdf['LON']]
gdf = gdf.drop(columns=['DIST2LAND', 'LON', 'LAT'])
gdf['time'] = pd.to_datetime(gdf['ISO_TIME'])
# %% Create and ERA5 API request for pressure and wind
import cdsapi
c = cdsapi.Client()

def get_era5_data(i, year, month, day, hour, lat, lon, eps=0.1):
    # NOTE: should get max lifetime wind rather than just the wind at the time of landfall
    if not os.path.exists(f'/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations/mangrove_damage_locations_{i}.nc'):
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure'
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': hour,
                'area': [
                    lat - eps, lon - eps, lat + eps, lon + eps
                ],
                'format': 'netcdf'
            },
            f'/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations/mangrove_damage_locations_{i}.nc'
        )
# %%
for i, row in gdf[['time', 'center_centerLat', 'center_centerLon']].iterrows():
    year = row['time'].year
    month = row['time'].month
    day = row['time'].day
    hour = str(row['time'].hour).zfill(2)
    hour = f"{hour}:00"
    lat = row['center_centerLat']
    lon = row['center_centerLon']
    get_era5_data(i, year, month, day, hour, lat, lon)
# %%
# load one of the files
import xarray as xr
ds = xr.open_dataset('/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations/_20.nc')
ds
# %%
import glob

def assign_index(x):
    x['index'] = x.encoding['source'].split('_')[-1].split('.')[0]
    return x

files = glob.glob('/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations/*.nc')
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4", preprocess=assign_index)


# %%
# files = glob.glob('/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations_subset/*.nc')
# ds_sub = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4", preprocess=assign_index)
# ds_sub['index']
# %%
ds['wind_speed'] = (ds['u10']**2 + ds['v10']**2)**0.5
del ds['u10']
del ds['v10']
# %%
df = ds.to_dataframe()
df = df.dropna()
df = df.reset_index()
# %%
!say done
# %%
df = df.groupby('index').agg({'latitude': 'first', 'longitude': 'first', 'msl': 'min', 'wind_speed': 'max'}).reset_index()
# %%
df = df.rename(columns={'latitude': 'center_centerLat',
                        'longitude': 'center_centerLon',
                        'msl': 'mslp',
                        'wind_speed': 'wind'})

# %%
# 'center_centerLat', 'center_centerLon', 'time'
merged = df[['index', 'mslp', 'wind']].join(gdf, on='index', how='inner', rsuffix='_era5')
# merged = pd.merge(gdf, df, on=['index'], how='left').dropna()
merged = merged.set_index('index').sort_values(by='index', ascending=True)
# %%
merged.to_csv("/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/era5_and_slope.csv")
# %%