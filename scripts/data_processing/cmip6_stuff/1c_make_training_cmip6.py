# %%
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# %%
wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = os.path.join(wd, "cmip6_data")

df = pd.read_csv(os.path.join(datadir, f"fitted_data.csv"))
coords = gpd.read_file(os.path.join(datadir, "coords.gpkg"))
df = df.merge(coords, on='grid')
df.columns = [col.replace('.', '_') for col in df.columns]
# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
gdf = gpd.GeoDataFrame(df, geometry='geometry').set_crs('EPSG:4326')
c0 = gdf['cluster'].min()
gdf[gdf['cluster'] == c0].plot(column='p_u10', legend=True, marker='s', cmap='viridis', ax=axs[0])
gdf[gdf['cluster'] == c0].plot(column='thresh_u10', legend=True, marker='s', cmap='viridis', ax=axs[1])
gdf[gdf['cluster'] == c0].plot(column='scale_u10', legend=True, marker='s', cmap='viridis', ax=axs[2])
gdf[gdf['cluster'] == c0].plot(column='shape_u10', legend=True, marker='s', cmap='viridis', ax=axs[3])
gdf[gdf['cluster'] == c0].plot('u10')
gdf.head()

# %%
# use latitude and longitude columns to label grid points in (i,j) format
gdf['latitude'] = gdf['geometry'].apply(lambda x: x.y)
gdf['longitude'] = gdf['geometry'].apply(lambda x: x.x)
gdf = gdf.sort_values(['latitude', 'longitude', 'cluster'], ascending=[True, True, True])

# %%
# get dimensions
channels = ['u10']
nchannels = len(channels)
T = gdf['cluster'].nunique()
nx = gdf['longitude'].nunique()
ny = gdf['latitude'].nunique()

# make training tensors
X = gdf[channels].values.T.reshape([nchannels, ny, nx, T])[:, ::-1, :, :]
U = gdf[[f"ecdf_{channel}" for channel in channels]].values.T.reshape([nchannels, ny, nx, T])[:, ::-1, :, :]
X = np.swapaxes(X, 0, -1)
U = np.swapaxes(U, 0, -1)
z = gdf[['cluster', 'extremeness_u10']].groupby('cluster').mean().values.reshape([T])
t = gdf[['cluster']].groupby('cluster').mean().reset_index().values.reshape([T])
plt.imshow(U[0, ..., 1])

# %%
# parameters and coordinates
mvars = ['u10', 'slp']
threshs = []
scales = []
shapes = []

gpd_params = [f"thresh_{var}" for var in mvars] + [f"scale_{var}" for var in mvars] + [f"shape_{var}" for var in mvars]
gdf_params = gdf[[*gpd_params, 'longitude', 'latitude']].groupby(['latitude', 'longitude']).mean().reset_index()
for var in mvars:
    threshs.append(gdf_params[f'thresh_{var}'].values.reshape([ny, nx])[::-1,...])
    scales.append(gdf_params[f'scale_{var}'].values.reshape([ny, nx])[::-1,...])
    shapes.append(gdf_params[f'shape_{var}'].values.reshape([ny, nx])[::-1,...])

thresh = np.stack(threshs, axis=0)
scale = np.stack(scales, axis=0)
shape = np.stack(shapes, axis=0)
params = np.stack([thresh, scale, shape], axis=-1)
lat = gdf_params['latitude'].values.reshape([ny, nx])[::-1,...]
lon = gdf_params['longitude'].values.reshape([ny, nx])[::-1,...]

# %%
np.savez(os.path.join(datadir, f"data.npz"), X=X, U=U, z=z, lat=lat, lon=lon, t=t, params=params)
# %%

