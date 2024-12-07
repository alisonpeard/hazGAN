# %%
import os
from environs import Env
import numpy as np
import xarray as xr

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str('DATADIR')
    mistral = env.str('MISTRAL')
    inpath = os.path.join(datadir, 'geo', 'etopo1', '1_arc_min', 'grd', 'ETOPO1_Bed_g_gdal.grd')
    outpath = os.path.join(mistral, 'hazGAN', 'constant-fields', 'etopo1.grd')

    ds = xr.open_dataset(inpath, engine="netcdf4")
    nx = 21601  # Longitude points (check)
    ny = 10801  # Latitude points
    z_2d = ds.z.values.reshape((ny, nx))
    z_2d = np.flip(z_2d, axis=0)


    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)

    # make new dataset for Bay of Bengal
    ds_2d = xr.Dataset(
        data_vars={
            'elevation': (['lat', 'lon'], z_2d)
        },
        coords={
            'lon': ('lon', lon),
            'lat': ('lat', lat)
        },
        attrs=ds.attrs
    )

    # save
    ds_2d = ds_2d.sel(lon=slice(80, 95), lat=slice(10, 25))
    ds_2d.to_netcdf(os.path.join(mistral, 'hazGAN', 'constant-fields', 'etopo1-bay-of-bengal.nc'))
    print("Saved NetCDF file.")

    # plot
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
        )
    
    ds_2d['elevation'].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='terrain',
        cbar_kwargs={'label': 'Elevation (m)'}
        
        )
    # map features
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title('ETOPO1 Elevation')

    # save
    figpath = os.path.join(mistral, 'hazGAN', 'etopo1-bay-of-bengal.png')
    fig.savefig(figpath, dpi=300, bbox_inches='tight')
    print('saved figure as', figpath)

    