# %%
# import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

bob_crs = 24346
xmin, xmax, ymin, ymax = 80, 95, 10, 25
# %%
slopepath = "/Users/alison/Documents/DPhil/data/athanisou-slopes-2019/nearshore_slopes.csv"
slope_df = pd.read_csv(slopepath)
slope_df
slope_df.describe()

# %%
slope_df = slope_df[(slope_df.Y >= ymin) & (slope_df.Y < ymax)]
slope_df = slope_df[(slope_df.X >= xmin) & (slope_df.X < xmax)]
# %%
slope_gdf = gpd.GeoDataFrame(slope_df, geometry=gpd.points_from_xy(slope_df.X, slope_df.Y)).set_crs(epsg=4326)
slope_gdf = slope_gdf.sort_values(by='slope', ascending=True)
slope_gdf.plot('slope', cmap='YlOrRd', markersize=2, legend=True)
# %%

bbox = box(xmin, ymin, xmax, ymax)
mangroves = gpd.read_file('/Users/alison/Documents/DPhil/data/gmw-v3-2020/gmw_v3_2020_vec.gpkg', mask=bbox)
mangroves = mangroves.set_crs(epsg=4326).drop(columns='PXLVAL')
mangroves = gpd.sjoin_nearest(mangroves.to_crs(bob_crs), slope_gdf[['slope', 'geometry']].to_crs(bob_crs), how='left', distance_col='distance').to_crs(4326)
# %% plot against patch centroids (polygons too small)
mangrove_centroids = mangroves.set_geometry(mangroves.centroid).sort_values(by='slope', ascending=True)
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy import feature


fig, axs = plt.subplots(1 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
slope_gdf.plot(ax=axs[0], column='slope', cmap='YlOrRd', legend=True)
mangrove_centroids.plot(ax=axs[1], column='slope', cmap='YlOrRd', legend=True)

for ax in axs:
    ax.add_feature(feature.LAND,facecolor='wheat') 
    ax.add_feature(feature.OCEAN)   
    ax.coastlines(resolution='50m',color='k', linewidth=.5) 
axs[0].set_title('Slope points [tan(beta)]')
axs[1].set_title('Nearest slope at mangrove patch centroid')
# %%
