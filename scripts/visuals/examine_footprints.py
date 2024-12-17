# %%
import os
import numpy as np
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt


FIELD = 0
RP_MIN, RP_MAX = (0.1, 0.5) # list[(0.1, 0.5), (0.5, 1), (1, 100)]


def main(datadir:str) -> None:
    """Plot the smallest and biggest storms in the dataset."""
    ds = xr.open_dataset(os.path.join(datadir, "data.nc"))
    ds = ds.where(ds.storm_rp > RP_MIN, drop=True)
    ds = ds.where(ds.storm_rp <= RP_MAX, drop=True)
    ds = ds.sortby('storm_rp', ascending=True)
    maxima = ds.isel(field=FIELD).anomaly.max(dim=['lon', 'lat'])

    # set colorbar to have center at zero
    vmax = ds.isel(field=FIELD, time=slice(0, 64))['anomaly'].max().values
    vmin = ds.isel(field=FIELD, time=slice(0, 64))['anomaly'].min().values
    cmap = plt.cm.Spectral_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # plot the smallest storms in the set
    fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, ax in enumerate(axs.ravel()):
        im = ds.isel(time=i).anomaly.isel(field=0).plot.contourf(
            ax=ax,
            levels=20,
            cmap=cmap,
            norm=norm,
            add_colorbar=False
            )
        ax.set_ylabel(f"{ds.isel(time=i).storm_rp.values:.2f}-year")
        ax.set_title("")
        ax.set_xlabel("")
        ax.label_outer()
    fig.colorbar(ax=axs, mappable=im, orientation='vertical', label='Wind anomaly', aspect=80)
    fig.suptitle(f"64 wind footprints with {RP_MIN} < RP < {RP_MAX} (smallest)", y=.95)


    # plot the biggest storms in the set
    ds = ds.sortby('storm_rp', ascending=False)
    vmax = ds.isel(field=FIELD, time=slice(0, 64))['anomaly'].max().values
    vmin = ds.isel(field=FIELD, time=slice(0, 64))['anomaly'].min().values
    cmap = plt.cm.Spectral_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, ax in enumerate(axs.ravel()):
        im = ds.isel(time=i).anomaly.isel(field=0).plot.contourf(
            ax=ax,
            levels=20,
            cmap=cmap,
            norm=norm,
            add_colorbar=False
            )
        ax.set_ylabel(f"{ds.isel(time=i).storm_rp.values:.2f}-year")
        ax.set_title("")
        ax.set_xlabel("")
        ax.label_outer()
    
    fig.colorbar(ax=axs, mappable=im, orientation='vertical', label='Wind anomaly', aspect=80)
    fig.suptitle(f"64 wind footprints with {RP_MIN} < RP < {RP_MAX} (biggest)", y=.95)
    print("Summary\n-------")
    print(f"Number of samples with {RP_MIN} < RP < {RP_MAX}: {ds.time.size}")
    print("Minimimum wind {:.2f}".format(maxima.min().values))
    print("Maximum wind {:.2f}".format(maxima.max().values))


# %%
if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")
    main(datadir)

# %%