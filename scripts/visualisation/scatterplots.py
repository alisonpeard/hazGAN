"""
Code to generate scatterplots of the samples for the highest, lowest, and middle EC pairs for 
training set, test set, Brown-Resnick, and deep learning samples.
"""
# %%
import hazGAN # stops things bugging out
# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from hazGAN import POT, scatter_density2
import matplotlib.pyplot as plt

channel = 0
channels = ['u10', 'mslp']

# %% 
# Load estimated extremal coefficients
ECs = pd.read_parquet("//Users/alison/Documents/DPhil/multivariate/results/brown_resnick/ECs.parquet")

# load Brown-Resnick samples
samples = pd.read_parquet(f"/Users/alison/Documents/DPhil/multivariate/results/brown_resnick/samples_{channels[channel]}.parquet")
coords = pd.read_parquet("/Users/alison/Documents/DPhil/multivariate/era5_data/coords.parquet")
coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.longitude, coords.latitude)).set_crs(4326)
coord_map = {grid: (lat, lon) for grid, lat, lon in zip(coords.grid, coords.latitude, coords.longitude)}
samples['lat'] = samples['grid'].map(lambda x: coord_map[x][0])
samples['lon'] = samples['grid'].map(lambda x: coord_map[x][1])
samples_idx = samples['grid'].values
samples = samples.set_index('grid')

# load train/test data
data = xr.open_dataset("/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc")
data = data.stack(grid=('lat', 'lon'))
ntrain = 1000
data_train = data.isel(time=slice(0, ntrain))
data_test = data.isel(time=slice(ntrain, None))
# %% sample observation points with high, median, and low ECs
ECs_OPs = ECs[ECs['i'].isin(samples_idx) & ECs['j'].isin(samples_idx)].reset_index(drop=False)
high_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmax()]
low_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmin()]
middle_EC_pair = ECs_OPs[(ECs_OPs['train_EC'] > 1.5) & (ECs_OPs['train_EC'] < 1.6)].sample(1).squeeze()

# %% PROCESS BROWN-RESNICK SAMPLES TO DATAFRAME ####################################################################
samples_U = samples.loc[:,'1':].values.swapaxes(0, 1)
samples_U = samples_U[..., np.newaxis]
params = data.isel(grid=samples_idx, channel=channel).params.values.swapaxes(0, 1) # annoying

#  make dataframe of full scale samples
X = data.isel(grid=samples_idx, channel=channel).anomaly.values[..., np.newaxis]
U = data.isel(grid=samples_idx, channel=channel).uniform.values[..., np.newaxis]
params = params[..., np.newaxis]
samples_PIT = POT.inv_probability_integral_transform(samples_U, X, U, params, gumbel_margins=False).squeeze()
samples_X = samples.reset_index(drop=False).loc[:, :'lat'].copy()
samples_X = pd.concat([samples_X, pd.DataFrame(samples_PIT.T, columns=np.arange(1, 1001).astype(str))], axis=1)
samples_X = samples_X.set_index('grid')
# %% SCATTER PLOTS ################################################################################################
def format_str(s):
    return s.replace('_', ' ').title()
# %%
i = high_EC_pair['i'].astype(int)
j = high_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
ax = axs[0]
x = data_train.isel(grid=i, channel=0).anomaly.values
y = data_train.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Training Set")

ax = axs[1]
x = data_test.isel(grid=i, channel=0).anomaly.values
y = data_test.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Test Set")

ax = axs[2]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick Samples")

for ax in axs:
    ax.set_ylim(-2, 10)
    ax.set_xlim(-2, 10)
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(high_EC_pair['train_EC']))
# %%
i = middle_EC_pair['i'].astype(int)
j = middle_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
ax = axs[0]
x = data_train.isel(grid=i, channel=0).anomaly.values
y = data_train.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Training Set")

ax = axs[1]
x = data_test.isel(grid=i, channel=0).anomaly.values
y = data_test.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Test Set")

ax = axs[2]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick Samples")

for ax in axs:
    ax.set_ylim(-2, 10)
    ax.set_xlim(-2, 10)
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(middle_EC_pair['train_EC']))
# %%
i = low_EC_pair['i'].astype(int)
j = low_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
ax = axs[0]
x = data_train.isel(grid=i, channel=0).anomaly.values
y = data_train.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Training Set")

ax = axs[1]
x = data_test.isel(grid=i, channel=0).anomaly.values
y = data_test.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Test Set")

ax = axs[2]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick Samples")

for ax in axs:
    ax.set_ylim(-2, 10)
    ax.set_xlim(-2, 10)
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(low_EC_pair['train_EC']))
# %%