# %%
import os
from environs import Env
import xarray as xr
import matplotlib.pyplot as plt

VISUALS = True

if __name__ == "__main__":
    # paths
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("DATADIR")
    outdir = env.str('TRAINDIR')
    inpath = os.path.join(datadir, "etopo1", "etopo1-bay-of-bengal.nc")
    outpath = os.path.join(outdir, "..", "constant-fields", "terrain.nc")

    # make land only
    ds = xr.open_dataset(inpath)
    land_only = ds.where(ds['elevation'] > 0, 0)
    land_only['mask'] = xr.where(ds['elevation'] > 0, 1, -1)
    land_only['normalised'] = (land_only['elevation'] - land_only['elevation'].min()) / (land_only['elevation'].max() - land_only['elevation'].min())
    
    if VISUALS:
        from cartopy import feature as cfeature
        import cartopy.crs as ccrs
        import matplotlib.colors as colors

        fig, axs = plt.subplots(1, 2, figsize=(12, 4),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        land_only['elevation'].plot(ax=axs[0], cmap="terrain", vmax=800,
                                    transform=ccrs.PlateCarree())
        axs[0].add_feature(cfeature.COASTLINE)
        axs[0].set_title("Elevation")

        land_only['mask'].plot(ax=axs[1], cmap="coolwarm", transform=ccrs.PlateCarree())
        axs[1].add_feature(cfeature.COASTLINE)
        axs[1].set_title("Land mask")

        fig.suptitle("ETOPO1 data")
    # save
    land_only.to_netcdf(outpath)
    print("Saved as", outpath)

    # save for Shruti's cGAN (optional)
    if False:
        projectpath = "/Users/alison/Documents/DPhil/github/downscaling-cgan/alison-data/constants"
        land_only['elevation'].to_netcdf(os.path.join(projectpath, 'elev.nc'))
        land_only['mask'].to_netcdf(os.path.join(projectpath, 'lsm.nc'))

# %%