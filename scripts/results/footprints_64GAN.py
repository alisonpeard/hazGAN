# %%
import os
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    config = {key: value["value"] for key, value in config.items()}
    return config

def calculate_total_return_periods(damages, yearly_rate, var='mangrove_damage_area', index='time'):
    totals = damages[var].max(dim=['lat', 'lon']).to_dataset()
    N = totals[var].sizes[index]
    totals['rank'] = totals[var].rank(dim=index)
    totals['exceedence_probability'] = 1 - ( totals['rank'] / ( N + 1 ) )
    totals['return_period'] = 1 / ( yearly_rate * totals['exceedence_probability'] )
    return totals['return_period']


res = (18, 22)
RUNNAME =  "serene-sweep-8" #"amber-sweep-13" # leafy-sweep-2"
datadir = f'/Users/alison/Documents/DPhil/paper1.nosync/training/{res[0]}x{res[1]}'
samplesdir = f'/Users/alison/Documents/DPhil/paper1.nosync/samples'
config = open_config(RUNNAME, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")
data = xr.open_dataset(os.path.join(datadir, "data.nc")).sel(channel=['u10', 'tp'])
samples_ds = xr.open_dataset(os.path.join(samplesdir, f"{RUNNAME}.nc"))
occurence_rate = data.attrs['yearly_freq']
samples_ds['storm_rp'] = calculate_total_return_periods(samples_ds.isel(channel=0), occurence_rate, var='anomaly', index='sample')
ds = samples_ds.rename({'sample': 'time'})

# %%
CHANNEL = 0
RP_lower, RP_upper = (0, 100)
OUTLIERS = np.array([], dtype='datetime64[ns]')

# %%
ds1 = ds.where(ds.storm_rp > RP_lower, drop=True)
ds1 = ds.where(ds1.storm_rp <= RP_upper, drop=True)
ds1 = ds1.where(~ds1.time.isin(OUTLIERS), drop=True)
ds1 = ds1.sortby('storm_rp', ascending=True)

# set colorbar to have center at zero
vmax = ds1.isel(channel=CHANNEL, time=slice(0, 64))['anomaly'].max().values
vmin = ds1.isel(channel=CHANNEL, time=slice(0, 64))['anomaly'].min().values
cmap = plt.cm.Spectral_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# plot the smallest
fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0, 'wspace': 0})
for i, ax in enumerate(axs.ravel()):
    im = ds1.isel(time=i).anomaly.isel(channel=0).plot.contourf(
        ax=ax,
        levels=20,
        cmap=cmap,
        norm=norm,
        add_colorbar=False
        )
    ax.set_ylabel(f"{ds1.isel(time=i).storm_rp.values:.2f}-year")
    ax.set_title("")
    ax.set_xlabel("")
    ax.label_outer()
fig.colorbar(ax=axs, mappable=im, orientation='vertical', label='Wind anomaly', aspect=80)
fig.suptitle(f"64 wind footprints with {RP_lower} < RP < {RP_upper} (smallest)", y=.95)


# plot the biggest
ds2 = ds1.sortby('storm_rp', ascending=False)
vmax = ds2.isel(channel=CHANNEL, time=slice(0, 64))['anomaly'].max().values
vmin = ds2.isel(channel=CHANNEL, time=slice(0, 64))['anomaly'].min().values
cmap = plt.cm.Spectral_r
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0, 'wspace': 0})
for i, ax in enumerate(axs.ravel()):
    im = ds2.isel(time=i).anomaly.isel(channel=0).plot.contourf(
        ax=ax,
        levels=20,
        cmap=cmap,
        norm=norm,
        add_colorbar=False
        )
    ax.set_ylabel(f"{ds1.isel(time=i).storm_rp.values:.2f}-year")
    ax.set_title("")
    ax.set_xlabel("")
    ax.label_outer()
fig.colorbar(ax=axs, mappable=im, orientation='vertical', label='Wind anomaly', aspect=80)
fig.suptitle(f"64 wind footprints with {RP_lower} < RP < {RP_upper} (biggest)", y=.95)
print(f"Number of samples with {RP_lower} < RP < {RP_upper}: {ds1.time.size}")

# %%