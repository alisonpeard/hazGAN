# %%
import os
import numpy as np
import xarray as xr
from environs import Env
import matplotlib
import matplotlib.pyplot as plt

import matplotlib 
mycmap = plt.get_cmap("PuBu")
darkest = matplotlib.colors.to_hex(mycmap.get_over())
markercolor = matplotlib.colors.to_hex(mycmap(0.5))
matplotlib.rc('axes', edgecolor=darkest)


env = Env()
env.read_env(recurse=True)
datadir = env.str("TRAINDIR")

data = xr.open_dataset(os.path.join(datadir, "64x64", "data_1941_2022.nc"))
data

# %%
import pandas as pd

tc_dates = pd.read_csv("/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/ibtracs_dates.csv")
tc_dates["time"] = pd.to_datetime(tc_dates["time"])
tc_dates = tc_dates.sort_values("time", ascending=False)
tc_dates = tc_dates[tc_dates['event'] != "Not_Named"]
tc_dates

# %%
storms = tc_dates.groupby("event").agg({"time": ["min", "max"], "wind": "max"})
storms.columns = ["start", "end", "wind"]
storms = storms.sort_values("start", ascending=False)
storms = storms.sort_values("wind", ascending=False)
nstorms = len(storms)

# %%
from scipy.ndimage import gaussian_filter

i = 4

storm = storms.iloc[i]
name  = storm.name
start = storm.start
end   = storm.end
wind  = storm.wind

# round start and end to nearest day
start = np.datetime64(pd.Timestamp(start).normalize()) 
end   = np.datetime64(pd.Timestamp(end).normalize())
start_buffer = start - np.timedelta64(30, 'D')
end_buffer   = end + np.timedelta64(30, 'D')

storm = data.where(data.time >= start_buffer, drop=True).where(data.time <= end_buffer, drop=True)
ndays = storm.time.size

# %%
markercolor = matplotlib.colors.to_hex(mycmap(0.5))

storm = storm.sortby("time")
fig, ax = plt.subplots(figsize=(30, 3))
storm.u10.max(dim=["lat", "lon"]).plot(ax=ax, color=darkest)
storm.u10.max(dim=["lat", "lon"]).plot(ax=ax, marker="o", color=markercolor, markersize=5)
# ax.axhline(y=15, color="crimson", linestyle="dashed")
# ax.fill_betweenx(ax.get_ylim(), start, end, color="crimson", alpha=0.2)
ax.axis("off")

# %%
start_buffer = start - np.timedelta64(3, 'D')
end_buffer   = end + np.timedelta64(2, 'D')

storm = data.where(data.time >= start_buffer, drop=True).where(data.time <= end_buffer, drop=True)
ndays = storm.time.size


storm_arr = storm.u10.values
storm_smoothed = gaussian_filter(storm_arr, sigma=2)
lats = storm.lat.values
lons = storm.lon.values
time = storm.time.values

storm = xr.DataArray(storm_smoothed, coords=[time, lats, lons], dims=["time", "lat", "lon"])
storm = storm.rename("u10")
storm = storm.sortby("time")

# %%
size  = 5
nlevels = 12
ncols = ndays
nrows = 1 # int(np.floor(ndays / ncols))
whitespace = 0.025



fig, axs = plt.subplots(1, ncols,
                        figsize=(
                            (size * ncols) + whitespace,
                            (size * nrows) + (nrows * whitespace)
                            ),
                        gridspec_kw={
                            "wspace": whitespace,
                            "hspace": whitespace
                        })

vmin = storm.min().values
vmax = storm.max().values
levels = np.linspace(vmin, vmax, nlevels)

for i, ax in enumerate(axs.flat):
    storm.isel(time=i).plot.contour(ax=ax, levels=levels,
                                        colors=darkest)
    storm.isel(time=i).plot.contourf(ax=ax, levels=levels, cmap=mycmap,
                                         alpha=1., add_colorbar=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.label_outer()

# fig.suptitle(f"Tropical Cyclone {name.title()}", fontsize=24);
print(name)

#%% make a matching time series
fig, ax = plt.subplots(1, 1,
                        figsize=(
                            (size * ncols) + whitespace,
                            ((size * nrows) + (nrows * whitespace)) * 0.8
                            ))

windmaxima = storm.max(dim=["lat", "lon"])

ax.bar(windmaxima.time.values, windmaxima.values, color=markercolor,
       alpha=0.5, edgecolor=darkest)
ax.axhline(y=np.quantile(windmaxima.values, 0.5), color="crimson", linestyle="dashed", linewidth=6)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
ax.set_yticks([]);

# %% FOOTPRINTS
footprint = storm.where(storm.time >= start, drop=True).where(storm.time <= end, drop=True)

for i in range(footprint.time.size):
    fig, ax = plt.subplots(figsize=(size, size))
    footprint.isel(time=i).plot.contour(ax=ax, levels=levels, colors=darkest)
    footprint.isel(time=i).plot.contourf(ax=ax, levels=levels, cmap=mycmap, alpha=1., add_colorbar=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.label_outer()
# %%
footprint = footprint.max(dim=["time"])

fig, ax = plt.subplots(figsize=(size, size))
footprint.plot.contour(ax=ax, levels=levels, colors=darkest)
footprint.plot.contourf(ax=ax, levels=levels, cmap=mycmap, alpha=1., add_colorbar=False)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")
ax.set_xticks([])
ax.set_yticks([])
ax.label_outer()

# %% 

