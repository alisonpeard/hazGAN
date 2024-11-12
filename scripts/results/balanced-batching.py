"""
View results from balanced-batching sweep.

Comments:
    celestial-sweep-6: max(u10)=80, no tracks
    gallant-sweep-18: max(u10)=120, no tracks
    pious-sweep-12: max(u10)=100, no tracks
"""
# %%
import os
import numpy as np
import xarray as xr
import hazGAN as hazzy
from environs import Env
import matplotlib.pyplot as plt

SWEEP = 'pious-sweep-12'
CHANNEL = 0

env = Env()
env.read_env(recurse=True)
wd = env.str('WORKINGDIR')
traindir = env.str('TRAINDIR')
indir = os.path.join(wd, "samples", "balanced-batching")

samples = np.load(os.path.join(indir, f"{SWEEP}.npz"))['uniform']
nchannels = samples.shape[-1]
print(nchannels, "channels.")

train = xr.open_dataset(os.path.join(traindir, 'data.nc'))
train = train.isel(channel=slice(0, nchannels))

# view samples in original space
help(hazzy.inv_probability_integral_transform)
T = hazzy.inv_probability_integral_transform
X = train['anomaly'].data
U = train['uniform'].data
params = train['params'].data
samples_scale = T(samples, X, U, params)
samples_scale = samples_scale[..., 0]

maxima = np.max(samples_scale, axis=(1, 2))
idxmax = np.argsort(maxima)[::-1][:64]

# %%
plottable = samples_scale.copy()
plottable = plottable[idxmax, ...]
fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = np.quantile(plottable, 0.005)
vmax = np.quantile(plottable, 0.995)
for i, ax in enumerate(axs.ravel()):
    sample = plottable[i]
    im = ax.imshow(sample, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    ax.invert_yaxis()
    ax.axis('off')
fig.colorbar(im, ax=axs.ravel().tolist())
# %%
plottable = samples_scale.copy()
idx = np.random.choice(range(1000), 64)
plottable = plottable[idx, ...]
fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = np.quantile(plottable, 0.005)
vmax = np.quantile(plottable, 0.995)
for i, ax in enumerate(axs.ravel()):
    sample = plottable[i]
    im = ax.imshow(sample, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    ax.invert_yaxis()
    ax.axis('off')
fig.colorbar(im, ax=axs.ravel().tolist())

# %%
