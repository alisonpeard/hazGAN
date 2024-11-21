# %%
import os
from environs import Env
import xarray as xr
import matplotlib.pyplot as plt

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
    land_only['elevation'].plot(cmap="terrain")
    # save
    land_only.to_netcdf(outpath)
    print("Saved as", outpath)

    # save for Shruti's cGAN (optional)
    if False:
        projectpath = "/Users/alison/Documents/DPhil/github/downscaling-cgan/alison-data/constants"
        land_only['elevation'].to_netcdf(os.path.join(projectpath, 'elev.nc'))
        land_only['mask'].to_netcdf(os.path.join(projectpath, 'lsm.nc'))

# %%