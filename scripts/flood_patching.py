#%%
import os
import numpy as np
import glob
import rioxarray as rio
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling

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
    scale_factor = 0.01 # upsample to save time while in dev --do this better once working
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
data = np.load('/Users/alison/Documents/DPhil/multivariate/era5_data/data.npz')
t = 1
U = data['U'][t, ..., 0].ravel()
lat = data['lat'].ravel()
lon = data['lon'].ravel()
quantiles = gpd.GeoDataFrame({'quantile': U}, geometry=gpd.points_from_xy(lon, lat)).set_crs(4326)
quantiles['return_period'] = 1 / (1 - quantiles['quantile']) # TODO: check
quantiles.plot('return_period')
# %%
import geospatial_utils as gu
agg = gu.grid_ckdnearest(gdfs, quantiles, ['return_period'])
agg = agg.reset_index()
agg.plot('return_period', legend=True)
# %%
