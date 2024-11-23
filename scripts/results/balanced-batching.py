"""
View results from balanced-batching sweep.

Comments:
    celestial-sweep-6: max(u10)=80, no tracks
    gallant-sweep-18: max(u10)=120, no tracks
    pious-sweep-12: max(u10)=100, no tracks
    cool-sweep-47:
        max(u1)=65, tracks, but limited and similar
    cool-sweep-47:
        max(u1)=85, tracks, but limited and similar
        kind of spotty artifacts
    silvery-sweep-62:
        max(u1)=71, tracks, but spotty again
    wandering-sweep-65:
        no tracks
    wild-sweep-56:
        no tracks
    rural-sweep-81: (best for now)


NOTE: all the best so far are wind-only or wind and pressure,
precipitation is having a negative effect.
"""
# %%
import os
import numpy as np
import xarray as xr
import hazGAN as hazzy
from environs import Env
import matplotlib.pyplot as plt

SWEEP = 'rural-sweep-81'
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
samples_scale = samples_scale[..., CHANNEL]

lats = np.linspace(10, 15, 18)
lons = np.linspace(80, 95, 22)
LON, LAT = np.meshgrid(lons, lats)
# %%
# plot Gumbel(0,1) samples
maxima = np.max(samples, axis=(1, 2, 3))
idxmax = np.argsort(maxima)[::-1][:64]

plottable = samples.copy()[..., CHANNEL]
plottable = np.take(plottable, idxmax, axis=0) # plottable[idxmax, ...]

fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = plottable.min() # p.quantile(plottable, 0.005)
vmax = plottable.max() #np.quantile(plottable, 0.995)
for i, ax in enumerate(axs.ravel()):
    sample = plottable[i, ...]
    im = ax.contourf(LON, LAT, sample, levels=15, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    ax.axis('off')
fig.colorbar(im, ax=axs.ravel().tolist())
fig.suptitle('Most extreme Gumbel samples', fontsize=24)

# %%
# plot largest samples
maxima = np.max(samples_scale, axis=(1, 2))
idxmax = np.argsort(maxima)[::-1][:64]

plottable = samples_scale.copy()
plottable = plottable[idxmax, ...]
fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = np.quantile(plottable, 0.005)
vmax = np.quantile(plottable, 0.995)
for i, ax in enumerate(axs.ravel()):
    sample = plottable[i]
    # im = ax.imshow(sample, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    im = ax.contourf(LON, LAT, sample, levels=15, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    ax.axis('off')
fig.colorbar(im, ax=axs.ravel().tolist())
fig.suptitle('Most extreme samples', fontsize=24)
# %%
# plot random samples
plottable = samples_scale.copy()
idx = np.random.choice(range(1000), 64)
plottable = plottable[idx, ...]
# %%
scale = True
fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = np.quantile(plottable, 0.005) if scale else None
vmax = np.quantile(plottable, 0.995) if scale else None
for i, ax in enumerate(axs.ravel()):
    sample = plottable[i]
    # im = ax.imshow(sample, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    im = ax.contourf(LON, LAT, sample, levels=15, vmin=vmin, vmax=vmax, cmap="Spectral_r")
    ax.axis('off')
if scale:
    fig.colorbar(im, ax=axs.ravel().tolist())
fig.suptitle('Random samples', fontsize=24)
# %%
# plot samples with max in [25, 35]
plottable = samples_scale.copy()
samples_max = np.max(plottable, axis=(1, 2))

# %%filter to resemble max training winds
condition = lambda x: (x >= 25) & (x <=40)
idx = np.argwhere(condition(samples_max)).squeeze()
print(len(idx))
plottable = plottable[idx, ...]
samples_max = samples_max[idx]
sorting = np.argsort(samples_max)[::-1]
plottable = plottable[sorting, ...]

# %%
scale = True
fig, axs = plt.subplots(8, 8, figsize=(16, 10))
vmin = -5 # np.quantile(plottable, 0.001)
vmax = 40 # np.quantile(plottable, 0.999)
for i, ax in enumerate(axs.ravel()):
    if i < len(plottable):
        sample = plottable[i]
        im = ax.contourf(LON, LAT, sample,
                        #  levels=15,
                        #  vmin=vmin, vmax=vmax,
                        levels=np.linspace(vmin, vmax, 15),
                        cmap="Spectral_r")
        ax.axis('off')
if scale:
    fig.colorbar(im, ax=axs.ravel().tolist())
fig.suptitle('Random samples', fontsize=24)
# %%