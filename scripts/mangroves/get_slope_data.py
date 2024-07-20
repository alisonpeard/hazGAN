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

indir = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2/result/model/'
outdir = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'

infile1 = os.path.join(indir, 'input_fixedCNTRY_rmOutlier.csv')
outfile1 = os.path.join(outdir, 'data_with_slopes.gpkg')
# %%
df = pd.read_csv(infile1)
[*df.columns]
df.landingYear.describe() # 2001 - 2017

# %% -----Add Athanisou's slope data------
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
slopepath = "/Users/alison/Documents/DPhil/data/athanisou-slopes-2019/nearshore_slopes.csv"
slope_df = pd.read_csv(slopepath)
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
gdf = gpd.sjoin_nearest(gdf.to_crs(bob_crs), slope_gdf[['slope', 'geometry']].to_crs(bob_crs), how='left', distance_col='distance').to_crs(4326)
gdf.sort_values(by='slope', ascending=True).plot('slope', cmap='YlOrRd', legend=True)

# %%
gdf.to_file(outfile1)
# %%
