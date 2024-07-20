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
from shapely.geometry import box
from hazGAN import xmin, xmax, ymin, ymax
bob_crs = 24346

indir1 = '/Users/alison/Documents/DPhil/data/athanisou-slopes-2019'
indir2 = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves')
infile1 = os.path.join(indir1, "nearshore_slopes.csv")
infile2 = os.path.join(indir2, 'mangroves.geojson')
outfile1 = os.path.join(indir2, 'mangroves_with_slope.geojson')
# %%
aoi = box(xmin, ymin, xmax, ymax)
slope_df = pd.read_csv(infile1)
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
slope_gdf = slope_gdf.clip(aoi).to_crs(bob_crs)

mangroves = gpd.read_file(infile2).to_crs(bob_crs)
mangroves_with_slope = mangroves.sjoin_nearest(slope_gdf, how='left')
mangroves_with_slope = mangroves_with_slope[['area', 'slope', 'geometry']].reset_index()
mangroves_with_slope = mangroves_with_slope.groupby('index').agg({'area': 'first', 'slope': 'mean', 'geometry': 'first'})
mangroves_with_slope = gpd.GeoDataFrame(mangroves_with_slope, geometry='geometry').set_crs(bob_crs)
mangroves_with_slope.plot('slope', cmap='YlOrRd', legend=True)
mangroves_with_slope.to_file(outfile1, driver='GeoJSON')
# %%
