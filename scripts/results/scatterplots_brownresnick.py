"""
Code to generate scatterplots of the samples for the highest, lowest, and middle EC pairs for 
training set, test set, Brown-Resnick, and deep learning samples.
"""
# %%
import hazGAN as hg # stops things bugging out
# %%
import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from hazGAN import POT, scatter_density2
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    config = {key: value["value"] for key, value in config.items()}
    return config

channel = 0
channels = ['u10', 'tp']
fig_kws = {'dpi': 300, 'bbox_inches': 'tight', 'transparent': True}

runname = 'amber-sweep-13'
datadir = '/Users/alison/Documents/DPhil/paper1.nosync/training/18x22'
samplesdir = f'/Users/alison/Documents/DPhil/paper1.nosync/samples'
resdir = '/Users/alison/Documents/DPhil/paper1.nosync/results/brown_resnick'
figdir = '/Users/alison/Documents/DPhil/paper1.nosync/figures/paper/brownresnick'

config = open_config(runname, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")

# %% -----Load data-----
data = xr.open_dataset(os.path.join(datadir, "data.nc"))
data = data.stack(grid=('lat', 'lon')).transpose('time', 'grid', 'param', 'channel').sel(channel=['u10', 'tp'])
ntrain = config['train_size']
data_train = data.isel(time=slice(0, ntrain))
data_test = data.isel(time=slice(ntrain, None))

# load hazGAN samples
samples_gan = xr.open_dataset(os.path.join(samplesdir, f"{runname}.nc"))
samples_gan = samples_gan.rename({'sample': 'time'}).stack(grid=('lat', 'lon'))
samples_gan = samples_gan.isel(time=slice(0, 1000))

# load Brown-Resnick samples
samples_br = pd.read_parquet(os.path.join(resdir, f"samples_{channels[channel]}.parquet"))
samples_idx = samples_br['grid'].values
samples_br = samples_br.set_index('grid')

# load estimated extremal coefficients
ECs = pd.read_parquet(os.path.join(resdir, f"ECs_{channels[channel]}.parquet"))

# sample observation points with high, median, and low ECs
ECs_OPs = ECs[ECs['i'].isin(samples_idx) & ECs['j'].isin(samples_idx)].reset_index(drop=False)
high_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmax()]
low_EC_pair = ECs_OPs.iloc[ECs_OPs['train_EC'].idxmin()]
low_EC_pair = ECs_OPs.sort_values(by='train_EC').iloc[2,:] # try second lowest
middle_EC_pair = ECs_OPs[(ECs_OPs['train_EC'] > 1.5) & (ECs_OPs['train_EC'] < 1.6)].sample(1).squeeze()

# %% -------Process Brown-Resnick samples to dataframe-------
samples_U = samples_br.loc[:,'1':].values.swapaxes(0, 1)
samples_U = samples_U[..., np.newaxis]
params = data.isel(grid=samples_idx, channel=channel).params.values # annoying

#  make dataframe of full scale samples
X = data_train.isel(grid=samples_idx, channel=channel).anomaly.values[..., np.newaxis]
U = data_train.isel(grid=samples_idx, channel=channel).uniform.values[..., np.newaxis]
params = params[..., np.newaxis]
samples_PIT = POT.inv_probability_integral_transform(samples_U, X, U, params, gumbel_margins=False).squeeze()
samples_X = samples_br.reset_index(drop=False).loc[:, :'lat'].copy()
samples_X = pd.concat([samples_X, pd.DataFrame(samples_PIT.T, columns=np.arange(1, 1001).astype(str))], axis=1)
samples_X = samples_X.set_index('grid')

# %% ------Uniform scatterplots------
def format_str(s):
    return s.replace('_', ' ').title()

pair = middle_EC_pair # ['high' | 'middle' | 'low']
pairs = [high_EC_pair, middle_EC_pair, low_EC_pair]
pair_names = ['high', 'middle', 'low']

for pair, name in zip(pairs, pair_names):
    xlim = (-.1, 1.1)
    ylim = (-.1, 1.1)

    i = pair['i'].astype(int)
    j = pair['j'].astype(int)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    ax = axs[0]
    x = data_train.isel(grid=i, channel=0).uniform.values
    y = data_train.isel(grid=j, channel=0).uniform.values
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("Training Set", fontsize=10)

    ax = axs[1]
    x = data_test.isel(grid=i, channel=0).uniform.values
    y = data_test.isel(grid=j, channel=0).uniform.values
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("Test Set", fontsize=10)

    ax = axs[2]
    x = samples_gan.isel(grid=i, channel=0).uniform.values
    y = samples_gan.isel(grid=j, channel=0).uniform.values
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("hazGAN", fontsize=10)

    ax = axs[-1]
    x = samples_br.loc[i, '1':].values.astype(float)
    y = samples_br.loc[j, '1':].values.astype(float)
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("Brown-Resnick", fontsize=10)

    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(format_str(samples_br.loc[i, 'obs_pt']))
        ax.set_ylabel(format_str(samples_br.loc[j, 'obs_pt']))
        ax.label_outer()

    fig.suptitle(r"$\hat \theta =${:.2f}".format(pair['train_EC']))
    fig.savefig(os.path.join(figdir, f"uniform_{channels[channel]}_{name}.png"), **fig_kws)

# %% -------Anomaly scatterplots-------
for pair, name in zip(pairs, pair_names):
    i = pair['i'].astype(int)
    j = pair['j'].astype(int)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
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
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    ax = axs[2]
    x = samples_gan.isel(grid=i, channel=0).anomaly.values
    y = samples_gan.isel(grid=j, channel=0).anomaly.values
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("hazGAN")

    ax = axs[-1]
    x = samples_X.loc[i, '1':].values.astype(float)
    y = samples_X.loc[j, '1':].values.astype(float)
    scatter_density2(x, y, cmap='magma', ax=ax)
    ax.set_title("Brown-Resnick")

    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(format_str(samples_br.loc[i, 'obs_pt']))
        ax.set_ylabel(format_str(samples_br.loc[j, 'obs_pt']))
        ax.label_outer()
    fig.suptitle(r"$\hat \theta =${:.2f}".format(pair['train_EC']))
    fig.savefig(os.path.join(figdir, f"anomaly_{channels[channel]}_{name}.png"), **fig_kws)

# %%