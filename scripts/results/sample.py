"""Generate and save an xarray of samples to use in other visualisations (save regenerating every time)"""
# %% quick settings
RUNNAME     = "neat-thunder-330" # first draft: amber-sweep-13
MAXRP       = 500
TEMPERATURE = 1 
CMAPS       = ["YlOrRd", "PuBu", "YlGnBu"]

rcParams['font.family'] = 'serif'
hist_kws = {'bins': 25, 'color': 'lightgrey', 'edgecolor': 'k', 'alpha': 0.5, 'density': True}

# paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
# %%
import os
import numpy as np
import xarray as xr
from environs import Env


import matplotlib.pyplot as plt
from matplotlib import rcParams

import hazGAN
from hazGAN import label_data
from hazGAN.torch import WGANGP
from hazGAN.torch import unpad
from hazGAN.constants import TEST_YEAR
from hazGAN.statistics import invPIT

# %%
env = Env()
env.read_env(recurse=True)
wd = env.str("WORKINGDIR")
figdir = env.str("IMAGEDIR")
os.chdir(os.path.join(wd, "hazGAN", "saved-models", RUNNAME))

config = hazGAN.load_config_from_yaml("config-defaults.yaml")
model = WGANGP(config, device='mps')
model.load_weights("checkpoint.weights.h5")

# ntrain = config.train_size

# %%
reference = xr.open_dataset(os.path.join(wd, "training", "18x22", "data.nc"))
reference = reference.sel(field=config['fields'])
train = reference.where(reference['time.year'] != TEST_YEAR, drop=True)
test = reference.where(reference['time.year'] == TEST_YEAR, drop=True)
# train = ds_ref.isel(time=slice(-ntrain, None))
# test = ds_ref.isel(time=slice(0, -ntrain))
occurrence_rate = reference.attrs['yearly_freq']
nsamples = int(occurrence_rate * MAXRP) # or 560

# %%
def sample_to_xr(data, reference, plot=False):
    nsamples = data.shape[0]
    samples = np.arange(nsamples)

    ds = xr.Dataset(
        {
        'uniform': (['sample', 'lat', 'lon', 'channel'], data),
        'params': (['lat', 'lon', 'param', 'channel'], reference.params.values),
    },
    coords={
        'sample': samples,
        'lat': ("lat", reference.lat.values, {"long_name": "Latitude"}),
        'lon': ("lon", reference.lon.values, {'long_name': 'Longitude'}),
        'channel': ['u10', 'tp'],
        'param': ("param", ['shape', 'loc', 'scale'], {'long_name': 'gpd_parameters'}),
        'month': np.unique(reference['time.month'].values)
        }
    )
    if plot:
        ds.isel(sample=0, channel=0).uniform.plot.contourf(levels=20, cmap='viridis')
    return ds

# %%
def batch_sample_wgan(labels, conditions, batch_size=100):
    nsamples = len(labels)
    samples = []
    for i in range(0, nsamples, batch_size):
        batch_size = min(batch_size, nsamples - i)
        # batch_samples = min(batch_size, nsamples - i)
        batch_labels = labels[i:i + batch_size]
        batch_conditions = conditions[i:i + batch_size]
        batch = unpad(model(label=batch_labels, condition=batch_conditions))
        batch = batch.detach().cpu().numpy()
        batch = batch.transpose(0, 2, 3, 1)
        samples.append(batch)
    samples = np.concatenate(samples, axis=0)
    return samples

# %% prepare for sampling
reference['maxwind'] = reference.sel(field='u10').anomaly.max(dim=['lat', 'lon'])
reference = reference.sortby('maxwind', ascending=False)

nsamples = min(nsamples, len(reference['maxwind'])) # in case we don't have enough data

conditions = reference['maxwind'].isel(time=slice(0, nsamples))
labels = label_data(conditions, config['thresholds'])
print("Counts [0, 1, 2]: ", list(np.bincount(labels)))

samples_hazGAN = batch_sample_wgan(labels, conditions)
ds_hazGAN = sample_to_xr(samples_hazGAN, reference, plot=True)

#%%  sample fully independent uniform data of same size
samples_independent = np.random.uniform(size=(nsamples, 18, 22, 2))
samples_dependent = np.random.uniform(size=(nsamples))
samples_dependent = np.repeat(samples_dependent, 18*22*2, axis=0).reshape(nsamples, 18, 22, 2)

# %% plot the sampled uniform values

fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True, sharex=True, layout='tight')
channel = 0
axs = axes[0, :]
axs[0].hist(samples_hazGAN[..., 0].flatten(), **hist_kws);
axs[1].hist(samples_independent[..., 0].flatten(), **hist_kws);
axs[2].hist(samples_dependent[..., 0].flatten(), **hist_kws);
axs[3].hist(train.uniform.values[..., 0].flatten(), **hist_kws);

axs[0].set_title('HazGAN')
axs[1].set_title('Independent')
axs[2].set_title('Dependent')
axs[3].set_title('Training')
axs[0].set_ylabel('Wind speed')

channel = 1
axs = axes[1, :]
axs[0].hist(samples_hazGAN[..., 1].flatten(), **hist_kws);
axs[1].hist(samples_independent[..., 1].flatten(), **hist_kws);
axs[2].hist(samples_dependent[..., 1].flatten(), **hist_kws);
axs[3].hist(train.uniform.values[..., 1].flatten(), **hist_kws);

axs[0].set_title('HazGAN')
axs[1].set_title('Independent')
axs[2].set_title('Dependent')
axs[3].set_title('Training')
axs[0].set_ylabel('Cumulative\nprecipitation')

for ax in axes.flatten():
    ax.axhline(1, color='r', linestyle='--', label='Target shape')
    ax.legend(loc='upper center', framealpha=1, edgecolor='k', fancybox=False)
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.label_outer()

fig.suptitle('Uniformity of pixels', fontsize=16)

# %% convert to original scale
sample_u = ds_hazGAN.uniform.values
train_x = train.anomaly.values
train_params = reference.params.values # note use reference not train

sample_x = invPIT(sample_u, train_x, train_params)
sample_ind = invPIT(samples_independent, train_x, train_params)
sample_dep = invPIT(samples_dependent, train_x, train_params)
ds_hazGAN['anomaly'] = (('sample', 'lat', 'lon', 'channel'), sample_x)
ds_hazGAN['independent'] = (('sample', 'lat', 'lon', 'channel'), sample_ind)
ds_hazGAN['dependent'] = (('sample', 'lat', 'lon', 'channel'), sample_dep)

# %% add monthly info
monthly_medians = reference.medians.groupby('time.month').median()
ds_hazGAN['medians'] = (('month', 'lat', 'lon', 'channel'), monthly_medians.values)
# %%
i = np.random.randint(0, nsamples)

ds_hazGAN.isel(sample=i,
               channel=0).anomaly.plot.contourf(
                   cmap='Spectral_r', levels=15
               )

# %%
ds_hazGAN = ds_hazGAN.rio.write_crs("EPSG:4326")
ds_hazGAN.to_netcdf(os.path.join(wd, "samples", f"{RUNNAME}.nc"), mode='w')
# %%
