# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
figdir = '/Users/alison/Documents/DPhil/paper1.nosync/figures/paper/fig1'

# ds = xr.open_dataset('/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/data_1950_2022.nc')
ds = xr.open_dataset('/Users/alison/Documents/DPhil/data/era5/bay_of_bengal__monthly.nosync/original/bangladesh_1951_07.nc')

# %%
fontsize = 20
n = 6
vmin = ds['u10'].quantile(0.01).values
vmax = ds['u10'].quantile(0.99).values

# fig, axs = plt.subplots(1, n, figsize=(int(4 * n), 3.1), gridspec_kw={'wspace': 0.02})
for i in range(n):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3.1))
    ds.isel(time=i).u10.plot(ax=ax, cmap='RdBu_r', add_colorbar=False, vmin=vmin, vmax=vmax)
    # ax.axis('off')
    ax.set_title(r"$\tilde X_{{t={}}}$".format(i+1), fontsize=fontsize)
    ax.set_xlabel('$j$', fontsize=fontsize)
    ax.set_ylabel('$i$', fontsize=fontsize)
    ax.set_xticks([])   
    ax.set_yticks([])
    plt.savefig(os.path.join(figdir, 'era5_u10', f'era5_u10_day{i}.png'), transparent=True)

# %% Plot daily maxima
n = 50
ts = ds.isel(time=slice(0, n)).u10.max(dim=['latitude', 'longitude'])
threshold = ts.quantile(0.6)
tmin, tmax = ts.time.min().values, ts.time.max().values

exceedences = ts.where(ts > threshold, drop=True)
maxima = ts.isel(time=[5, 27, 32, 46])

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.vlines(ts.time, 0, ts, color='k', lw=0.5, label='Daily maxima')
ax.axhline(threshold, color='blue', lw=1, linestyle='--', label='Threshold')

ax.vlines(exceedences.time, 0, exceedences, color='blue', lw=1, label="Storms")
ax.scatter(maxima.time, maxima, color='red', s=20, marker='x', label='Storm maxima')
ax.legend(loc='lower left', fontsize='x-small', facecolor='white', framealpha=1, fancybox=False)

xticks = ax.get_xticks()
xticklabels = [f'{t}' for t in np.linspace(0, n, len(xticks), dtype=int)]
ax.set_xticks(xticks, labels=xticklabels, rotation=0);
ax.set_xlabel('$t$')
ax.set_ylabel('$M_t$ (m/s)')
plt.tight_layout()
plt.savefig(os.path.join(figdir, f'daily_maxima.png'), transparent=True)

# %% sample footprints
storm1 = ds.isel(time=slice(0, 9)).u10.max(dim=['time'])
storm2 = ds.isel(time=27).u10#.max(dim=['time'])
storm3 = ds.isel(time=slice(30, 34)).u10.max(dim=['time'])
storm4 = ds.isel(time=slice(45, 47)).u10.max(dim=['time'])
storms = [storm1, storm2, storm3, storm4]

for i, storm in enumerate(storms):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.1))
    storm1.plot(ax=ax, cmap='RdBu_r', add_colorbar=False, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(f'Storm {i+1}')
    plt.savefig(os.path.join(figdir, 'storms_u10', f'era5_u10_storm{i}.png'), transparent=True)
                 
# %% known storm footprints