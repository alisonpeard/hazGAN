"""Generate and save an xarray of samples to use in other visualisations (save regenerating every time)"""
# %%
import os
import xarray as xr
import tensorflow as tf
import hazGAN as hg
from hazGAN import POT
import wandb
# %%
wd = "/Users/alison/Documents/DPhil/multivariate/hazGAN"
RUNNAME = "new-dimensions"
os.chdir(os.path.join(wd, "saved-models", RUNNAME))
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
cmaps = ["YlOrRd", "PuBu", "YlGnBu"]
figdir = "/Users/alison/Documents/DPhil/multivariate/hazGAN/figures/results"
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)
wgan.generator.load_weights(os.path.join(wd, "saved-models", RUNNAME, f"generator.weights.h5"))
wgan.generator.summary()
ntrain = config.train_size
# %%
import numpy as np
ds_ref = xr.open_dataset(os.path.join(wd, "..", "era5_data", "data.nc"))
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
        'lat': ds_ref.lat.values,
        'lon': ds_ref.lon.values,
        'channel': ['u10', 'mslp'],
        'param': ['shape', 'loc', 'scale']
        }
        )
    if plot:
        ds.isel(sample=0, channel=0).uniform.plot.contourf(levels=10, cmap='viridis')
    return ds

samples_GAN = hg.unpad(wgan(nsamples=1000), paddings).numpy()
ds_GAN = sample_to_xr(samples_GAN, ds_ref, plot=True)

# %% convert to original scale
sample_U = ds_GAN.uniform.values
X = ds_ref.isel(time=slice(0, ntrain)).anomaly.values
U = ds_ref.isel(time=slice(0, ntrain)).uniform.values
sample_X = POT.inv_probability_integral_transform(sample_U, X, U, ds_ref.isel(time=slice(0, ntrain)).params.values)
ds_GAN['anomaly'] = (('sample', 'lat', 'lon', 'channel'), sample_X)
# %% add monthly info
monthly_medians = ds_ref.medians.groupby('time.month').median()
ds_GAN['median'] = monthly_medians
# %%
i = np.random.randint(0, 1000)
ds_GAN.isel(sample=i, channel=0).anomaly.plot(levels=10, cmap='viridis')
# %%
ds_GAN.to_netcdf(os.path.join(wd, "..", "era5_data", "hazGAN_samples.nc"))
# %%
