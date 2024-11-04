"""
Create a pre-training dataset for the GAN as follows:
-----------------------------------------------------
    1. Create sliding windows of u10 and tp for 2 to 20 day windows, covering all the 
    lengths of storms in the storm footprint data.
    2. Deseasonalise using monthly medians
    3. Transform to Gumbel(0, 1) using the empirical CDF.
    4. Save as a new xarray dataset.

Not implemented:
----------------
    - Removing outliers / wind and precip bombs
    - Accounting for autocorrelations 
"""
# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
datadir = "/Users/alison/Documents/DPhil/paper1.nosync/training"
res = (22, 18)

ds = xr.open_dataset(os.path.join(datadir, f"{res[1]}x{res[0]}", "data_1950_2022.nc"))
u10 = ds.u10.values.flatten()
plt.hist(u10, bins=100)
plt.yscale('log')

# %% look at event duration histograms to select window length(s)
ds_train = xr.open_dataset('/Users/alison/Documents/DPhil/paper1.nosync/training/18x22/data.nc')
ds_train['duration'].plot.hist()
window_length = [2, 5, 8, 10, 12, 15, 20]

# %% deseasonalise (with medians)
monthly = ds.groupby('time.month').median()
ds = ds.groupby('time.month') - monthly

# %%
from hazGAN import sliding_windows

window_length = [2, 5, 8]
window_length = [2, 5, 8, 10, 12, 15, 20]
                 
u10s = []
tps = []
msls = []
windows = []
for window in window_length:
    u10 = sliding_windows(ds['u10'].values, window)
    tp = sliding_windows(ds['tp'].values, window)
    msl = sliding_windows(ds['msl'].values, window)
    window_length = [window] * u10.shape[0]

    # reduce arrays along window dimensions
    u10 = u10.max(axis=1)
    tp = tp.sum(axis=1)
    msl = msl.sum(axis=1)

    # append to list
    u10s.append(u10)
    tps.append(tp)
    msls.append(msl)
    windows.append(window_length)

u10 = np.concatenate(u10s, axis=0)
tp = np.concatenate(tps, axis=0)
msl = np.concatenate(msls, axis=0)
window_length = np.concatenate(windows, axis=0)

X = np.stack([u10, tp, msl], axis=-1)
X.shape

# %% make a new xarray dataset
ds_window = xr.Dataset(
    {
        'anomaly': (('time', 'latitude', 'longitude', 'channel'), X),
        'window_length': (('time',), window_length)
    },
    coords={
        'time': range(u10.shape[0]),
        'latitude': ds.latitude,
        'longitude': ds.longitude,
        'channel': ['u10', 'tp', 'mslp']
        }
)
ds = ds_window

# %% Transform to Gumbel(0, 1) using the empirical CDF
def ecdf(ds, var, index_var='time'):
    rank = ds[var].rank(dim=index_var, keep_attrs=True)
    ecdf = rank / (len(ds[index_var]) + 1)
    return ecdf

def gumbel(ds, var):
    uniform = ecdf(ds, var)
    return -np.log(-np.log(uniform))

ds['uniform'] = ecdf(ds, 'anomaly')

# %% Have a look
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
ds.isel(longitude=0, latitude=0, channel=0).anomaly.plot.hist(ax=axs[0, 0])
ds.isel(longitude=0, latitude=0, channel=0).uniform.plot.hist(ax=axs[0, 1])
ds.isel(time=100, channel=0).anomaly.plot.contourf(ax=axs[1, 0], levels=20, cmap='Spectral_r')
ds.isel(time=100, channel=0).uniform.plot.contourf(ax=axs[1, 1], levels=20, cmap='Spectral_r')

# %% Save to netCDF and NumPy
ds.to_netcdf(os.path.join(datadir, f"{res[1]}x{res[0]}", "data_pretrain.nc"))
np.savez(os.path.join(datadir, f"{res[1]}x{res[0]}", "data_pretrain.npz"), data=X)

# %%
def notify(title, subtitle, message):
    os.system("""
              osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
              """.format(message, title, subtitle))
notify("Process finished", "Python script", "Finished making pretraining data")

# %%
