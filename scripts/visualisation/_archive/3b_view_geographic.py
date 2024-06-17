# %%
import os
import numpy as np
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
# %% load netcdf data
ds = xr.open_dataset("/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc")
ds.isel(time=0, channel=0).uniform.plot()
# %%
gdf = ds.to_dataframe().reset_index()
gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.lon, gdf.lat))
gdf = gdf.set_crs(4326)
gdf.total_bounds
# %%
# area of interest: array([80., 10., 95., 25.])
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
ops_gdf.head()

# %%
import pandas as pd
coords = pd.read_parquet("/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet")
coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.longitude, coords.latitude)).set_crs(4326)
# %%
ops = gpd.sjoin_nearest(ops_gdf.to_crs(3857), coords.to_crs(3857), how="inner")[['grid', 'index_right', 'latitude', 'longitude']]
ops = gpd.GeoDataFrame(ops, geometry=gpd.points_from_xy(ops.longitude, ops.latitude))[['grid', 'geometry']].set_crs(4326)
# %%
import matplotlib.pyplot as plt
u10 = gdf[gdf['channel'] == 'u10']
u10 = u10[u10['time'] == ds.time[0].values]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ds.isel(time=0, channel=0).uniform.plot(ax=ax, cmap="YlOrRd", vmin=0, vmax=1)
ops.plot(ax=ax, color="blue", alpha=.8, marker="o", markersize=50, label='Nearest grid centroids')
ops_gdf.plot(ax=ax, marker='^', color="black", label='Observation points')
for x, y, label in zip(ops.geometry.x, ops.geometry.y, ops.index):
    ax.annotate(label.title().replace('_', ':'), xy=(x, y), xytext=(3, 3), textcoords="offset points")
ax.legend()
ax.set_title('Observation points and grid centroids')
#%%
ops['lon'] = ops.geometry.x
ops['lat'] = ops.geometry.y
ops = ops.drop(columns=['geometry']).reset_index(drop=False)
ops = ops.rename(columns={'index': 'obs_pt'})
ops[['obs_pt', 'grid', 'lon', 'lat']].to_parquet("/Users/alison/Documents/DPhil/multivariate/era5_data/ops.parquet")
# %% train/test ECs
ECs = pd.read_parquet("//Users/alison/Documents/DPhil/multivariate/results/brown_resnick/ECs.parquet")
# lets use this to choose a high EC pair, a middle EC pair, and a low EC pair
# high EC pair
high_EC = ECs[ECs['train_EC'] > 1.9].sample(1)
# middle EC pair
middle_EC = ECs[(ECs['train_EC'] > 1.5) & (ECs['train_EC'] < 1.6)].sample(1)
# low EC pair
low_EC = ECs[ECs['train_EC'] < 1.1].sample(1)

# %%
ECs_train = np.zeros((396, 396))
ECs_test = np.zeros((396, 396))
for _, row in ECs.iterrows():
    ECs_train[int(row['i'] - 1), int(row['j'] - 1)] = row['train_EC']
    ECs_test[int(row['i'] - 1), int(row['j'] - 1)] = row['test_EC']

ECs_train += ECs_train.T
ECs_test += ECs_test.T
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(ECs_train, cmap="Spectral")
im = axs[1].imshow(ECs_test, cmap="Spectral")
plt.colorbar(im, ax=axs)
# %% Brown-Resnick samples
channel = 0
channels = ['u10', 'mslp']
samples = pd.read_parquet(f"/Users/alison/Documents/DPhil/multivariate/results/brown_resnick/samples_{channels[channel]}.parquet")
coord_map = {grid: (lat, lon) for grid, lat, lon in zip(coords.grid, coords.latitude, coords.longitude)}
samples['lat'] = samples['grid'].map(lambda x: coord_map[x][0])
samples['lon'] = samples['grid'].map(lambda x: coord_map[x][1])
# %% # scatter two 
import xarray as xr

data = xr.open_dataset("/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc")
ntrain = 1000
grid_idx = samples['grid'].values

#%% match params to grid numbers
from hazGAN import POT

params = data.params.values[..., channel].reshape(18*22, 3)[grid_idx, ...]
X = data.X.values[:ntrain, ..., channel].reshape(18*22, ntrain)[grid_idx, ...]
U = data.U.values[:ntrain, ..., channel].reshape(18*22, ntrain)[grid_idx, ...]

sample_arr = samples.loc[:,'1':].values
sample_arr = sample_arr.T[..., np.newaxis]
X = X.T[..., np.newaxis]
U = U.T[..., np.newaxis]
params = params[..., np.newaxis]
result = POT.inv_probability_integral_transform(sample_arr, X, U, params, gumbel_margins=False).squeeze()
# %%
samples_X = samples.loc[:, :'lat'].copy()
samples_X = pd.concat([samples_X, pd.DataFrame(result.T, columns=np.arange(1, 1001).astype(str))], axis=1).set_index('obs_pt')

# %% look at Pearson correlation between observation points
samples_transpose = samples_X.T.loc['1':, :]
samples_transpose.corr()
# %% make heatmap of correlations
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(samples_transpose.corr(), ax=ax, cmap='magma')

# %%






