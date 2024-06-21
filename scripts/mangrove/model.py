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
# %% ------Train a model------
# TODO: pressure, precipitation
regressors = ['slope', 'landingWindLocal2', 'stormFrequency_mean', 'elevation_mean',
              'totalPrec_total', 'coastDist']
response = ['intensity']