# %%
import os
import numpy as np
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.pyplot as plt

figdir = '/Users/alison/Documents/DPhil/paper1.nosync/figures/paper/brownresnick'
ds = xr.open_dataset("/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/data.nc")
fig_kws = {'dpi': 300, 'bbox_inches': 'tight', 'transparent': True}

# create a variable grid, which is the index of each grid cell from 1-396
grid = np.arange(0, 396).reshape(18, 22) # may need to modify this
ds['grid'] = (('lat', 'lon'), grid)

observation_points = {
    # bangladesh
    'chittagong': (91.8466, 22.3569),
    'dhaka': (90.4125, 23.8103),
    'khulna': (89.5911, 22.8456),
    # mayanmar
    'sittwe': (92.9000, 20.1500),
    #'akyab': (92.9000, 20.1500),
    'rangoon': (96.1561, 16.8409),
    # india
    'kolkata': (88.3639, 22.5726),
    'madras': (80.2707, 13.0827),
    #'chennai': (80.2707, 13.0827),
    'vishakapatham': (83.3165, 17.6868),
    'haldia': (87.9634, 22.0253),
    # noaa buoys
    'buoy_23223': (89.483, 17.333),
    'buoy_23009': (90.0, 15.0),
    'buoy_23219': (88.998, 13.472),
    'buoy_23008': (90.0, 12.0),
    'buoy_23218': (88.5, 10.165),
    #'buoy_23401': (88.55, 8.86),
    'buoy_23007': (90.0, 8.0)
}
ops_gdf = gpd.GeoDataFrame.from_dict(observation_points, orient='index', columns=['lon', 'lat'], geometry=gpd.points_from_xy(*zip(*observation_points.values())))
ops_gdf = ops_gdf.set_crs(4326)
indices = []
for i, op in ops_gdf.iterrows():
    lon, lat = op.lon, op.lat
    indices.append(ds.sel(lat=lat, lon=lon, method='nearest').grid.values.item())
ops_gdf['grid'] = indices

# %% -----Plot observation points-----
from adjustText import adjust_text

fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(linewidth=.5)
ax.add_feature(feature.BORDERS, linestyle=':', linewidth=.2)
ax.add_feature(feature.LAND, color='tan')
ax.add_feature(feature.OCEAN, color='lightblue')

ops_gdf.plot(ax=ax, marker='o', color='red', edgecolor='k', markersize=100, zorder=100)

texts = []
for x, y, label in zip(ops_gdf.geometry.x, ops_gdf.geometry.y, ops_gdf.index):
    texts.append(ax.text(x, y, label.title().replace('_', ':'), fontsize=12, color='k', fontweight='bold'))

    # ax.annotate(label.title().replace('_', ':'), xy=(x, y), xytext=(3, 3),
    #             textcoords="offset points", fontsize=12, color='k')
# fix the above so that labels don't overlap

adjust_text(texts, expand=(1.5, 1.5), arrowprops=dict(arrowstyle="-", color='k', lw=0.2))
fig.savefig(os.path.join(figdir, 'observation_points.png'), transparent=True)

# %%