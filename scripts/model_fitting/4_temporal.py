# %%
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import hazardGANv2 as hg

datadir = os.path.join("..", "..", "era5_data")
data = np.load(os.path.join(datadir, "data.npz"))
event_data = pd.read_csv(os.path.join(datadir, "event_data.csv"))
fitted_data = pd.read_csv(os.path.join(datadir, "fitted_data.csv"))
era5 = pd.read_csv( "/Users/alison/Documents/DPhil/multivariate/era5_data/data_1950_2012.csv")
coords = gpd.read_file(os.path.join(datadir, "coords.gpkg"))
# %%
# let's start with the first event
i = event_data["cluster"][0]
event = event_data[event_data["cluster"] == i].copy()
t0 = event["time"].min()
t1 = event["time"].max()
era5 = era5[(era5["time"] >= t0) & (era5["time"] <= t1)].copy()
era5 = era5.merge(coords[["grid", "latitude", "lonsgitude"]], on="grid")
#
# %%
era5 = era5.sort_values(by=["latitude", "longitude", "time"])
# %%
T = era5["time"].nunique()
X = era5[["u10", "msl"]].values.T.reshape(2, 18, 22, T)[:, ::-1, :, :]
X = np.swapaxes(X, 0, -1)
plt.imshow(X[0, ..., 1])
# %%
fig, axs = plt.subplots(2, T + 1, figsize=(12, 3))

maxima = data['X'][0]
medians = data['M'][0]

u10 = maxima[..., 0] + medians[..., 0]
mslp = medians[..., 1] - maxima[..., 1]
vmin0 = np.min(u10)
vmax0 = np.max(u10)
vmin1 = np.min(mslp)
vmax1 = np.max(mslp)

im = axs[0, -1].imshow(u10)
plt.colorbar(im)
im = axs[1, -1].imshow(maxima[..., 1])
plt.colorbar(im)
for t in range(T):
    axs[0, t].imshow(X[t, :, :, 0], vmin=vmin0, vmax=vmax0)
    axs[0, t].set_title(era5["time"].unique()[t])
for t in range(T):
    im = axs[1, t].imshow(medians[:, :, 1])#, vmin=vmin1, vmax=vmax1)
    plt.colorbar(im)

# %%
