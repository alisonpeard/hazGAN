# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

datadir = "/Users/alison/Documents/DPhil/paper1.nosync/training/18x22"
ds = xr.open_dataset(os.path.join(datadir, "data.nc"))
ds

# %%
CHANNEL = 0
RP_lower, RP_upper = (1, 100)
OUTLIERS = np.array([
    '1992-04-15T00:00:00.000000000',
    '1952-05-09T00:00:00.000000000',
    '1995-05-02T00:00:00.000000000'
    ], dtype='datetime64[ns]')

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