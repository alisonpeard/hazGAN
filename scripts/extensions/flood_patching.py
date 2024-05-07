#%%
# env: general
import os
import numpy as np
import glob
import rioxarray as rio
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

rp_map = {'rp0250': 250,
          'rp0050': 500,
          'rp0025': 25,
          'rp0002': 2,
          'rp0000': 0,
          'rp0010': 10,
          'rp0005': 5,
          'rp0100': 100}

# %% Process dataframe of flood hazard maps
flood_dir = '/Users/alison/Documents/DPhil/data/deltares-bgd'
ref_year = 2018
files = glob.glob(os.path.join(flood_dir, '*.tif'))
files = [file for file in files if str(ref_year) in file]
rps = [file.split('_')[-1].split('.')[0] for file in files]
gdfs = None
cols = []
for file, rp in zip(files, rps):
    image = rio.open_rasterio(file)
    scale_factor = 0.1 # upsample to save time while in dev --do this better once working
    image = rio.open_rasterio(file)
    res = image.rio.resolution()
    image = image.rio.reproject(image.rio.crs, resolution=(res[0] / scale_factor, res[1] / scale_factor), resampling=Resampling.nearest)
    coords = gpd.read_file('/Users/alison/Documents/DPhil/multivariate/era5_data/coords.gpkg')
    df = image.to_dataframe(name='flood').reset_index()
    df = df[['y', 'x', 'flood']]
    df.columns = ['lat', 'lon', 'flood']
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']))
    gdf = gdf.set_crs(4326)
    gdf = gdf.replace(np.nan, 0)
    gdf = gdf[gdf['flood'] < 100] # TODO: quickfix do better later
    if gdfs is None:
        gdfs = gdf[['lat', 'lon', 'flood']].copy()
        gdfs.columns = ['lat', 'lon', f'flood_{rp}']
    else:
        gdfs = pd.merge(gdfs, gdf[['lat', 'lon', 'flood']], on=['lat', 'lon'], suffixes=('', f'_{rp}'))
    cols.append(f'flood_{rp}')
gdfs = gpd.GeoDataFrame(gdfs, geometry=gpd.points_from_xy(gdfs['lon'], gdfs['lat'])).set_crs(4326)
gdfs.sort_values(by='flood_rp0250', ascending=True).plot('flood_rp0250', cmap='Spectral_r')
gdfs.to_file('/Users/alison/Documents/DPhil/multivariate/deltares_data/flood_hazard.gpkg', driver='GPKG')

# %% Load uniform marginals sample from training data
from shapely.geometry import box

data = np.load('/Users/alison/Documents/DPhil/multivariate/era5_data/data.npz')
U = data['U'].reshape((data['U'].shape[0], 18 * 22 * 2))
U_flat = np.mean(U, axis=1)
tmax = np.argmax(U_flat)
# %%
t = tmax
U = data['U'][t, ..., 0].ravel()
lat = data['lat'].ravel()
lon = data['lon'].ravel()
quantiles = gpd.GeoDataFrame({'quantile': U}, geometry=gpd.points_from_xy(lon, lat)).set_crs(4326)
quantiles['return_period'] = 1 / (1 - quantiles['quantile']) # TODO: check
fig, ax = plt.subplots()
quantiles.plot('return_period', legend=True, ax=ax)
aoi = gpd.GeoDataFrame([], geometry=[box(*gdf.total_bounds)]).set_crs(4326)
aoi.boundary.plot(ax=ax, color='red')
# %%
import geospatial_utils as gu
agg = gu.grid_ckdnearest(gdfs, quantiles, ['return_period'], k=1, aggfunc=max)
sample = gdfs.join(agg['return_period'])
sample.plot('return_period', legend=True)
# %%
def interpolate_row(row, rp_map):
    """
    Logarithmic scaling between return periods.
    
    §JBA Global Flood Model Technical Report.
    """
    r_y = row['return_period']
    rps = list(rp_map.values())
    r_l = max([r for r in rps if r <= r_y])
    r_u = min([r for r in rps if r > r_y])
    d_l = row[f'flood_rp{str(r_l).zfill(4)}']
    d_u = row[f'flood_rp{str(r_u).zfill(4)}']
    print('lower:', r_l, 'upper:', r_u, 'd_l:', d_l, 'd_u:', d_u, 'r_y:', r_y)
    eps = 1e-6
    return d_l + (d_u - d_l) * (np.log(r_y+eps) - np.log(r_l+eps)) / (np.log(r_u+eps) - np.log(r_l+eps))


# %%
sample['d_y'] = sample.apply(interpolate_row, axis=1, args=(rp_map,))
sample = sample.replace(0, np.nan)
sample = sample.sort_values(by='d_y', ascending=True)
# %%
cmap = plt.cm.get_cmap('Blues', 6)
fig, axs = plt.subplots(1, 4, figsize=(18, 3))
sample.plot('return_period', legend=True, cmap='Reds', ax=axs[0], vmin=0, vmax=250)
sample.plot('return_period', legend=True, cmap='Reds', ax=axs[1], vmin=0, vmax=250)
sample.plot('return_period', legend=True, cmap='Reds', ax=axs[2], vmin=0, vmax=250)
sample.plot('return_period', legend=True, cmap='Reds', ax=axs[3], vmin=0, vmax=250)
sample.sort_values(by='flood_rp0000', ascending=True).plot('flood_rp0000', legend=True, cmap=cmap, ax=axs[1], vmin=0, vmax=5, alpha=.8)
sample.plot('d_y', legend=True, cmap=cmap, ax=axs[2], vmin=0, vmax=5, alpha=.8)
sample.sort_values(by='flood_rp0250', ascending=True).plot('flood_rp0250', legend=True, cmap=cmap, ax=axs[3], vmin=0, vmax=5, alpha=.8)

axs[0].set_title('Pixelwise return periods')
axs[1].set_title('0-year return period')
axs[2].set_title('Interpolated return period')
axs[3].set_title('250-year return period')

for ax in axs:
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.label_outer()
# %%
sample['diff'] = sample['flood_rp0250'] - sample['flood_rp0002']
sample.plot('diff', legend=True, cmap='Spectral_r')
# %%
