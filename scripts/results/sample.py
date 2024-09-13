"""Generate and save an xarray of samples to use in other visualisations (save regenerating every time)"""
# %%
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import hazGAN as hg
from hazGAN import POT
import wandb

# %%
wd = "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN"
RUNNAME = "soft-sweep-12" # "absurd-sweep-1" # "atomic-sweep-1" # "amber-sweep-13"
TEMPERATURE = 1 #+ 1e-2
os.chdir(os.path.join(wd, "saved-models", RUNNAME))
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
nyears = 500 # what return period do we want to go up to, breaks down after 242 right now

cmaps = ["YlOrRd", "PuBu", "YlGnBu"]
figdir = "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/figures/results"
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)
# wgan.generator.load_weights(os.path.join(wd, "saved-models", RUNNAME, f"generator.weights.h5"))
wgan.load_weights(os.path.join(wd, "saved-models", RUNNAME, "checkpoint.weights.h5"))
wgan.generator.summary()
ntrain = config.train_size

# %%
ds_ref = xr.open_dataset(os.path.join(wd, "..", "training", "18x22", "data.nc")).sel(channel=['u10', 'tp'])
train = ds_ref.isel(time=slice(0, ntrain))
test = ds_ref.isel(time=slice(ntrain, None))
occurrence_rate = ds_ref.attrs['yearly_freq']
nsamples = int(occurrence_rate * nyears)
# nsamples = 560
# %% Temp: playing with data sizes
nsamples = 560
ntrain = 324
ntest = 560 - ntrain
print(nsamples, ntrain, ntest)

# sort ds_ref by storm_rp
ds_ref = ds_ref.sortby('storm_rp', ascending=False)
ds_ref = ds_ref.isel(time=slice(0, nsamples))
ds_ref = ds_ref.sortby('time')
ds_train = ds_ref.isel(time=slice(0, ntrain))
ds_test = ds_ref.isel(time=slice(ntrain, None))

# plot examples from train and test
plt.hist(ds_train.storm_rp, bins=20, alpha=0.5, label='train')
plt.hist(ds_test.storm_rp, bins=20, alpha=0.5, label='test')
plt.yscale('log')
plt.legend();


# %%
def sample_to_xr(data, ds_ref, plot=False):
    nsamples = data.shape[0]
    samples = np.arange(nsamples)

    ds = xr.Dataset(
        {
        'uniform': (['sample', 'lat', 'lon', 'channel'], data),
        'params': (['lat', 'lon', 'param', 'channel'], ds_ref.params.values),
    },
    coords={
        'sample': samples,
        'lat': ("lat", ds_ref.lat.values, {"long_name": "Latitude"}),
        'lon': ("lon", ds_ref.lon.values, {'long_name': 'Longitude'}),
        'channel': ['u10', 'tp'],
        'param': ("param", ['shape', 'loc', 'scale'], {'long_name': 'gpd_parameters'}),
        'month': np.unique(ds_ref['time.month'].values)
        }
    )
    if plot:
        ds.isel(sample=0, channel=0).uniform.plot.contourf(levels=20, cmap='viridis')
    return ds


def batch_sample_wgan(nsamples, batch_size, temp):
    samples = []
    for i in range(0, nsamples, batch_size):
        batch_samples = min(batch_size, nsamples - i)
        batch = hg.unpad(wgan(nsamples=batch_samples, temp=temp, seed=i), paddings).numpy()
        samples.append(batch)
    samples = np.concatenate(samples, axis=0)
    return samples

# %%
samples_hazGAN = batch_sample_wgan(nsamples, 100, TEMPERATURE)
ds_hazGAN = sample_to_xr(samples_hazGAN, train, plot=True)

#  sample fully independent uniform data of same size
samples_independent = np.random.uniform(size=(nsamples, 18, 22, 2))
samples_dependent = np.random.uniform(size=(nsamples))
samples_dependent = np.repeat(samples_dependent, 18*22*2, axis=0).reshape(nsamples, 18, 22, 2)

# %% sanity check the totals from WGAN samples
u10_totals = samples_hazGAN.sum(axis=(1, 2))[..., 0]
u10_totals = np.unique(u10_totals)
print(len(u10_totals))

# %% plot the sampled uniform values
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
hist_kws = {'bins': 25, 'color': 'lightgrey', 'edgecolor': 'k', 'alpha': 0.5, 'density': True}

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
    ax.legend(loc='lower center', framealpha=1, edgecolor='k', fancybox=False)
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.label_outer()

# fig.tight_layout()
fig.suptitle('Uniformity of pixels', fontsize=16)

# %% convert to original scale
sample_U = ds_hazGAN.uniform.values
X = train.anomaly.values
U = train.uniform.values
sample_X = POT.inv_probability_integral_transform(sample_U, X, U, ds_ref.isel(time=slice(0, ntrain)).params.values)
sample_ind = POT.inv_probability_integral_transform(samples_independent, X, U, ds_ref.isel(time=slice(0, ntrain)).params.values)
sample_dep = POT.inv_probability_integral_transform(samples_dependent, X, U, ds_ref.isel(time=slice(0, ntrain)).params.values)
ds_hazGAN['anomaly'] = (('sample', 'lat', 'lon', 'channel'), sample_X)
ds_hazGAN['independent'] = (('sample', 'lat', 'lon', 'channel'), sample_ind)
ds_hazGAN['dependent'] = (('sample', 'lat', 'lon', 'channel'), sample_dep)
# %% add monthly info
monthly_medians = ds_ref.medians.groupby('time.month').median()
ds_hazGAN['medians'] = (('month', 'lat', 'lon', 'channel'), monthly_medians.values)
# %%
i = np.random.randint(0, nsamples)
ds_hazGAN.isel(sample=i, channel=0).anomaly.plot(cmap='viridis') # levels=10, 
# %%
ds_hazGAN = ds_hazGAN.rio.write_crs("EPSG:4326")
ds_hazGAN.to_netcdf(os.path.join(wd, "..", "samples", f"{RUNNAME}.nc"), mode='w')
# %%
