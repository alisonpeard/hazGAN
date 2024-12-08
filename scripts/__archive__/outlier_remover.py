""""
Newest outlier processing as of evening 06-12-2024
"""
# %% load the environment
import os
import glob
import numpy  as np
import pandas as pd
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt

THRESHOLD = 0.8 # to start

def frobenius(test:np.ndarray, template:np.ndarray) -> float:
    similarity = np.sum(template * test) / (np.linalg.norm(template) * np.linalg.norm(test))
    return similarity


env = Env()
env.read_env(recurse=True)  # read .env file, if it exists
datadir = env.str('TRAINDIR')

# load data and process
ds = xr.open_dataset(os.path.join(datadir, "data_1950_2022_raw.nc"))
ds['maxwind'] = ds['u10'].max(dim=['latitude', 'longitude'])

# %% sort by maximum wind
ds = ds.sortby('maxwind', ascending=False)
fig, axs = plt.subplots(8, 8)
for i, ax in enumerate(axs.ravel()):
    ds.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
    ax.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
plt.show()

#
# %%
# TODO: make if not exists ifelse
def rescale(x:np.array) -> np.array:
    return (x - x.min() / (x.max() - x.min()))

template = ds.isel(time=0).u10.data
template = rescale(template)
similarities = [] * ds['time'].data.size
for i in range(ds['time'].data.size):
    test_matrix = ds['u10'].isel(time=i).data.copy()
    test_matrix = rescale(test_matrix)
    similarity = frobenius(test_matrix, template)
    similarities.append(similarity)
similarities = np.array(similarities)

# %% plot in order of similarity
order = np.argsort(similarities)[::-1]
ds_ordered = ds.isel(time=order).copy()

# plot the first 400
fig, axs = plt.subplots(20, 20)
for i, ax in enumerate(axs.ravel()):
    ds_ordered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
    ax.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
plt.show()

# %% now mask to remove wind bombs
threshold = 0.98 #! need to do manual bisection to choose
print(f'{sum(similarities > threshold)} ERA5 "wind bombs" detected in dataset for threshold {threshold}.')
mask = similarities <= threshold

ds_filtered = ds.isel(time=mask)
ds_filtered = ds_filtered.sortby('maxwind', ascending=False)
ds_filtered = ds_filtered.isel(time=slice(0, 60000))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ds['u10'].isel(time=0).plot(ax=axs[0])
ds_filtered['u10'].isel(time=0).plot(ax=axs[1])

# %%
# plot the first 400
fig, axs = plt.subplots(20, 20)
for i, ax in enumerate(axs.ravel()):
    ds_filtered.isel(time=i).u10.plot(ax=ax, add_colorbar=False)
    ax.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
plt.show()

# %% now save and stuff
ds_filtered.to_netcdf(os.path.join(datadir, "data_1950_2022.nc"))
# %%
