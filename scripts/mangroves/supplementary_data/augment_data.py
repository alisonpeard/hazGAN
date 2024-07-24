"""
Secondary script to run with ERA5 API data
"""
#%%
import os
import xarray as xr
import pandas as pd
import geopandas as gpd
bob_crs = 24346
# %%
path = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2/input_fixedCNTRY_rmOutlier.csv'
df = pd.read_csv(path)
[*df.columns]
# %%
subset = df[df['stormName'] == 'Alex']
subset = subset[subset['mangrovePxl'] == 17]
subset
# %% -----Add Athanisou's slope data------
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
slopepath = "/Users/alison/Documents/DPhil/data/athanisou-slopes-2019/nearshore_slopes.csv"
slope_df = pd.read_csv(slopepath)
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
gdf = gpd.sjoin_nearest(gdf.to_crs(bob_crs), slope_gdf[['slope', 'geometry']].to_crs(bob_crs), how='left', distance_col='distance').to_crs(4326)
gdf.sort_values(by='slope', ascending=True).plot('slope', cmap='YlOrRd', legend=True)

# %%
# # load one of the files
ds = xr.open_dataset('/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations_subset/mangrove_damage_locations_YASI_97.nc')
ds
# %%
import glob

def assign_index(x):
    x['index'] = x.encoding['source'].split('_')[-1].split('.')[0]
    return x

files = glob.glob('/Users/alison/Documents/DPhil/data/era5/mangrove_damage_locations_subset/*.nc')
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4", preprocess=assign_index)
df = ds.to_dataframe()
df = df.dropna()
df = df.reset_index()
df = df.groupby('index').agg({'latitude': 'first', 'longitude': 'first', 'mslp': 'min', 'wind': 'max'}).reset_index()

# %%
df['index'] = df['index'].astype(int)
merged['landingWindMaxLocal2'] *= 1000 / 3600
merged = df[['index', 'mslp', 'wind']].join(gdf, on='index', how='inner', rsuffix='_era5')
# merged = pd.merge(gdf, df, on=['index'], how='left').dropna()
merged = merged.set_index('index').sort_values(by='index', ascending=True)
merged[['mslp', 'landingPressure', 'wind', 'landingWindMaxLocal2']]
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2, figsize=(10, 5))
merged.hist('wind', bins=10, ax=axs[0], color='lightgrey', edgecolor='k')
merged.hist('landingWindMaxLocal2', bins=10, ax=axs[1], color='lightgrey', edgecolor='k')
# %%
merged.to_csv("/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/era5_max_and_slope.csv")
# %%