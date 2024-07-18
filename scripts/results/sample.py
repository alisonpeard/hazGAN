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
RUNNAME = "clean-sweep-3"  # "toasty-serenity-21"
os.chdir(os.path.join(wd, "saved-models", RUNNAME))
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
occurrence_rate = 18.033
nyears = 200 # what return period do we want to go up to, breaks down after 242 right now
nsamples = int(occurrence_rate * nyears)

cmaps = ["YlOrRd", "PuBu", "YlGnBu"]
figdir = "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/figures/results"
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)
wgan.generator.load_weights(os.path.join(wd, "saved-models", RUNNAME, f"generator.weights.h5"))
wgan.generator.summary()
ntrain = config.train_size
# %%
ds_ref = xr.open_dataset(os.path.join(wd, "..", "training", "res_18x22", "data.nc"))
train = ds_ref.isel(time=slice(0, ntrain))
test = ds_ref.isel(time=slice(ntrain, None))

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
        'channel': ['u10', 'mslp'],
        'param': ("param", ['shape', 'loc', 'scale'], {'long_name': 'gpd_parameters'}),
        'month': np.unique(ds_ref['time.month'].values)
        }
    )
    if plot:
        ds.isel(sample=0, channel=0).uniform.plot.contourf(levels=20, cmap='viridis')
    return ds

samples_hazGAN = hg.unpad(wgan(nsamples=nsamples), paddings).numpy()
ds_hazGAN = sample_to_xr(samples_hazGAN, train, plot=True)
# %% sample fully independent uniform data of same size
samples_independent = np.random.uniform(size=(nsamples, 18, 22, 2))
samples_dependent = np.random.uniform(size=(nsamples))
samples_dependent = np.repeat(samples_dependent, 18*22*2, axis=0).reshape(nsamples, 18, 22, 2)

# %% plot the sampled uniform values
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(25, 5))
axs[0].hist(samples_hazGAN.flatten(), bins=100, color='blue', alpha=0.5, label='hazGAN', density=True)
axs[1].hist(samples_independent.flatten(), bins=100, color='red', alpha=0.5, label='independent', density=True);
axs[2].hist(samples_dependent.flatten(), bins=100, color='green', alpha=0.5, label='dependent', density=True);
axs[3].hist(train.uniform.values.flatten(), bins=100, color='purple', alpha=0.5, label='reference', density=True);
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
ds_hazGAN.to_netcdf(os.path.join(wd, "..", "samples", f"{RUNNAME}.nc"))
# %%
