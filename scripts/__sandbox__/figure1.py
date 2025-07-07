# %%
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# %%

# bbox = [80, 10, 95, 25]
# region = box(*bbox)
# region = gpd.GeoDataFrame(geometry=[region], crs='EPSG:4326')

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(2,2))
# region.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
ax.set_extent([80, 95, 10, 25])
ax.add_feature(cfeature.LAND.with_scale('110m'), color="#F5F5F5")
ax.add_feature(cfeature.OCEAN.with_scale('110m'), color="#DAE8FC")
ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.8, edgecolor='#666666')
plt.axis('off')
plt.tight_layout()

fig.savefig('/Users/alison/Desktop/roi.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
# %%
import xarray as xr
datapath = "/Users/alison/Documents/DPhil/paper1.nosync/training/64x64/data.nc"
ds = xr.open_dataset(datapath)

# %%
ds['maxwind'] = ds.isel(field=0).anomaly.max(dim=['lon', 'lat'])
ds = ds.sortby('maxwind', ascending=False) 

# %%
time =ds.time[0].data
# %%

ds.isel(field=0, time=0).anomaly.plot(cmap="Spectral_r")
# %%
