"""
Plot samples with different return periods.
"""
#%%
import os 
import numpy as np
import yaml
import xarray as xr
import hazGAN as hg
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams['font.family'] = 'serif'
hist_kws = {'bins': 25, 'color': 'lightgrey', 'edgecolor': 'k', 'alpha': 0.5, 'density': True}

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    config = {key: value["value"] for key, value in config.items()}
    return config

# %%
res = (18, 22)
RUNNAME =  "amber-sweep-13" #"amber-sweep-13" # leafy-sweep-2"
datadir = f'/Users/alison/Documents/DPhil/paper1.nosync/training/{res[0]}x{res[1]}'
samplesdir = f'/Users/alison/Documents/DPhil/paper1.nosync/samples'
config = open_config(RUNNAME, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")
data = xr.open_dataset(os.path.join(datadir, "data.nc")).sel(channel=['u10', 'tp'])
# data = data.where(data.storm_rp > 1, drop=True) # NOTE: this is for testing
samples_ds = xr.open_dataset(os.path.join(samplesdir, f"{RUNNAME}.nc"))

occurence_rate = 560 / 72 #data.attrs['yearly_freq'], if using top 560 only
ntrain = config['train_size']
# samples_ds = samples_ds.isel(sample=slice(0, ntrain)) # maybe fairer?
train_ds = data.isel(time=slice(-ntrain, None))
test_ds = data.isel(time=slice(0, -ntrain))
samples_ds = samples_ds.rename({'sample': 'time'})
print(train_ds.time.size, test_ds.time.size, samples_ds.time.size)

# %% ----Plot test set footprints with different return periods----
CHANNEL = 0
TIME = 0
RANDOM_TIME = lambda x: np.random.randint(0, x.time.size, 1)[0]
DATASET = train_ds
DATASET['variable'] = DATASET['anomaly'] + DATASET['medians']

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(DATASET['storm_rp'].values, **hist_kws)
ax.set_yscale('log')    

return_periods = [0, 1, 5, 10, 40]
fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.storm_rp > lower) & (DATASET.storm_rp <= upper), drop=True)
    n = samples.time.size
    t = RANDOM_TIME(samples)
    samples = samples.isel(time=t, channel=CHANNEL)
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year [{t}]")
    ax[i-1].set_xlabel(f"Number of samples: {n}")

fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.storm_rp > lower) & (DATASET.storm_rp <= upper), drop=True)
    samples = samples.isel(channel=CHANNEL).mean(dim='time')
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year")

fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.storm_rp > lower) & (DATASET.storm_rp <= upper), drop=True)
    samples = samples.isel(channel=CHANNEL).std(dim='time')
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year")

# %% ----Plot the four highest RP footprints----
# 1 -> 5 shows what GAN is making
# 5 -> 40 shows nice cyclones
bounds = [(1, 5), (5, 10), (10, 40)]
for lower, upper in bounds:
    samples = DATASET.where((DATASET.storm_rp > lower) & (DATASET.storm_rp <= upper), drop=True)
    n = min(samples.time.size, 5)

    fig, axs = plt.subplots(1, n, figsize=(20, 3))
    for i, ax in enumerate(axs):
        samples_i = samples.isel(time=i, channel=CHANNEL)
        samples_i.variable.plot.contourf(ax=ax, cmap='Spectral_r', add_colorbar=True, levels=20)
        ax.set_title(f"1-in-{int(samples_i.storm_rp.values)} year")

# %% ----Plot GAN footprints with different return periods----
MONTH = 7
CHANNEL = 0
TIME = 0

def calculate_total_return_periods(damages, yearly_rate, var='mangrove_damage_area', index='time'):
    totals = damages[var].max(dim=['lat', 'lon']).to_dataset()
    N = totals[var].sizes[index]
    totals['rank'] = totals[var].rank(dim=index)
    totals['exceedence_probability'] = 1 - ( totals['rank'] / ( N + 1 ) )
    totals['return_period'] = 1 / ( yearly_rate * totals['exceedence_probability'] )
    return totals['return_period']

samples_ds['return_period'] = calculate_total_return_periods(samples_ds.isel(channel=CHANNEL), occurence_rate, var='anomaly', index='time')

rps = samples_ds['return_period'].values
plt.hist(rps, **hist_kws)
plt.yscale('log')
plt.xlabel('Return period')
plt.ylabel('Frequency (log)')

# return_periods = [0, 25, 50, 250, 500]
return_periods = [0, 1, 5, 10, 40]

medians = samples_ds['medians'].isel(month=MONTH)
samples_ds['variable'] = samples_ds['anomaly'] + medians
DATASET = samples_ds.isel(channel=CHANNEL)
fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.return_period > lower) & (DATASET.return_period <= upper), drop=True)
    n = samples.time.size
    samples = samples.isel(time=RANDOM_TIME(samples))
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year")
    ax[i-1].set_xlabel(f"Number of samples: {n}")
fig.suptitle('Random samples from GAN footprints', y=1.05)

fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.return_period > lower) & (DATASET.return_period <= upper), drop=True)
    samples = samples.mean(dim='time')
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year")
fig.suptitle('Mean GAN footprints', y=1.05)

fig, ax = plt.subplots(1, len(return_periods)-1, figsize=(20, 3))
for i in np.arange(1, len(return_periods)):
    lower = return_periods[i-1]
    upper = return_periods[i]
    samples = DATASET.where((DATASET.return_period > lower) & (DATASET.return_period <= upper), drop=True)
    samples = samples.std(dim='time')
    samples.variable.plot.contourf(ax=ax[i-1], cmap='Spectral_r', add_colorbar=True, levels=20)
    ax[i-1].set_title(f"1-in-{upper} year")
fig.suptitle('Std GAN footprints', y=1.05)
# %% 
