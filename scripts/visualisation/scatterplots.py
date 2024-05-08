"""
Code to generate scatterplots of the samples for the highest, lowest, and middle EC pairs for 
training set, test set, Brown-Resnick, and deep learning samples.
"""
# %%
import hazGAN as hg # stops things bugging out
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
ds_ref = data.copy(deep=True) # for constructing other dataset
data = data.stack(grid=('lat', 'lon')).transpose('time', 'grid', 'param', 'channel')
ntrain = 1000
data_train = data.isel(time=slice(0, ntrain))
data_test = data.isel(time=slice(ntrain, None))

# %% sample observation points with high, median, and low ECs
ECs_OPs = ECs[ECs['i'].isin(samples_idx) & ECs['j'].isin(samples_idx)].reset_index(drop=False)
high_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmax()]
low_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmin()]
low_EC_pair = ECs_OPs.sort_values(by='train_EC').iloc[2,:] # try second lowest
middle_EC_pair = ECs_OPs[(ECs_OPs['train_EC'] > 1.5) & (ECs_OPs['train_EC'] < 1.6)].sample(1).squeeze()

# %% hazGAN SAMPLES ################################################################################################
import os
import tensorflow as tf
import wandb

wd = "/Users/alison/Documents/DPhil/multivariate/hazGAN"
RUNNAME = "sample-run"
os.chdir(os.path.join(wd, "saved-models", RUNNAME))
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
cmaps = ["YlOrRd", "PuBu", "YlGnBu"]
figdir = "/Users/alison/Documents/DPhil/multivariate/hazGAN/figures/results"
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)
wgan.generator.load_weights(os.path.join(wd, "saved-models", RUNNAME, f"generator.weights.h5"))
wgan.generator.summary()
# %%
def sample_to_xr(data, ds_ref, plot=False):
    ds = ds_ref.isel(time=slice(0, ntrain)).copy(deep=True)
    ds = ds.drop_vars(['anomaly', 'medians', 'extremeness', 'duration'])
    ds = ds.rename_dims({'time': 'sample'})
    ds['uniform'] = (('sample', 'lat', 'lon', 'channel'), data)
    if plot:
        ds.isel(sample=0, channel=0).uniform.plot()
    return ds

samples_GAN = hg.unpad(wgan(nsamples=ntrain), paddings).numpy()
ds_GAN = sample_to_xr(samples_GAN, ds_ref, plot=True)
sample_U = ds_GAN.uniform.values
X = ds_ref.isel(time=slice(0, ntrain)).anomaly.values
U = ds_ref.isel(time=slice(0, ntrain)).uniform.values
sample_X = POT.inv_probability_integral_transform(sample_U, X, U, ds_ref.isel(time=slice(0, ntrain)).params.values)
ds_GAN['anomaly'] = (('sample', 'lat', 'lon', 'channel'), sample_X)
ds_GAN = ds_GAN.stack(grid=('lat', 'lon')).transpose('sample', 'grid', 'param', 'channel')

# %% PROCESS BROWN-RESNICK SAMPLES TO DATAFRAME ####################################################################
samples_U = samples.loc[:,'1':].values.swapaxes(0, 1)
samples_U = samples_U[..., np.newaxis]
params = data.isel(grid=samples_idx, channel=channel).params.values # annoying

#  make dataframe of full scale samples
X = data_train.isel(grid=samples_idx, channel=channel).anomaly.values[..., np.newaxis]
U = data_train.isel(grid=samples_idx, channel=channel).uniform.values[..., np.newaxis]
params = params[..., np.newaxis]
samples_PIT = POT.inv_probability_integral_transform(samples_U, X, U, params, gumbel_margins=False).squeeze()
samples_X = samples.reset_index(drop=False).loc[:, :'lat'].copy()
samples_X = pd.concat([samples_X, pd.DataFrame(samples_PIT.T, columns=np.arange(1, 1001).astype(str))], axis=1)
samples_X = samples_X.set_index('grid')
# %% SCATTER PLOTS ################################################################################################
def format_str(s):
    return s.replace('_', ' ').title()
# %% low correlation
i = high_EC_pair['i'].astype(int)
j = high_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 4, figsize=(10, 4))
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
x = ds_GAN.isel(grid=i, channel=0).anomaly.values
y = ds_GAN.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("hazGAN")

ax = axs[-1]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick")

for ax in axs:
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(high_EC_pair['train_EC']))
# %% medium correlation
i = middle_EC_pair['i'].astype(int)
j = middle_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 4, figsize=(10, 4))
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
x = ds_GAN.isel(grid=i, channel=0).anomaly.values
y = ds_GAN.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("hazGAN")

ax = axs[-1]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick")

for ax in axs:
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(middle_EC_pair['train_EC']))
# %%
i = low_EC_pair['i'].astype(int)
j = low_EC_pair['j'].astype(int)

fig, axs = plt.subplots(1, 4, figsize=(10, 4))
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
x = ds_GAN.isel(grid=i, channel=0).anomaly.values
y = ds_GAN.isel(grid=j, channel=0).anomaly.values
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("hazGAN")

ax = axs[-1]
x = samples_X.loc[i, '1':].values.astype(float)
y = samples_X.loc[j, '1':].values.astype(float)
scatter_density2(x, y, cmap='magma', ax=ax)
ax.set_title("Brown-Resnick")

for ax in axs:
    ax.set_xlabel(format_str(samples.loc[i, 'obs_pt']))
    ax.set_ylabel(format_str(samples.loc[j, 'obs_pt']))
    ax.label_outer()
fig.suptitle(r"$\hat \theta =${:.2f}".format(low_EC_pair['train_EC']))
# %%