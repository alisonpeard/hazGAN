# %%
import os
from environs import Env
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
# %%
env = Env()
env.read_env(recurse=True)
traindir = env.str("TRAINDIR")
os.listdir(traindir)
# %%
ds = xr.open_dataset(os.path.join(traindir, 'data.nc'))
ds = ds.where(ds.storm_rp > 1., drop=True)
ds = ds.sel(channel='u10')
ds.isel(time=0).anomaly.plot()

# %%

outdir = os.path.join(traindir, 'images', "uniform")

nimgs = ds.time.size
for i in range(nimgs):
    array = ds.isel(time=i).uniform.values
    array = np.flip(array, axis=0)
    img = Image.fromarray(np.uint8(array * 255), 'L')
    img.save(os.path.join(outdir, f"storm_{i}.png"))

# %%
outdir = os.path.join(traindir, 'images', "anomaly")

nimgs = ds.time.size
for i in range(nimgs):
    array = ds.isel(time=i).anomaly.values
    array = np.flip(array, axis=0)
    plt.imshow(array);plt.show()
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min)
    img = Image.fromarray(np.uint8(array * 255), 'L')
    img.save(os.path.join(outdir, f"storm_{i}.png"))
    
# %%
