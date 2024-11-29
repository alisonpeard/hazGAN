"""
Create a pre-training dataset for the GAN as follows:
-----------------------------------------------------
    1. Create sliding windows of u10 and tp for 2 to 20 day windows, covering all the 
    lengths of storms in the storm footprint data.
    2. Deseasonalise using monthly medians
    4. Transform to uniform using the empirical from the storm data (since
        that's the data we are trying to supplement).
    5. Save as a new xarray dataset.

Not implemented:
----------------
    - Accounting for autocorrelations 

    
Input files:
-----------
    - data_1950_2022.nc | source: scripts/data_processing/process_resampled.py
    - data.nc | source: scripts/data_processing/make_training.py
Output files:
------------
    - data_pretrain.nc

"""
# %%
import os
from environs import Env
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from datetime import datetime
import subprocess, os

from hazGAN import sliding_windows


RESOLUTION = (22, 18)
VISUALISATIONS = False
WINDOWS = [2, 5, 8, 10, 12, 15, 20]


def notify(title, subtitle, message):
    os.system("""
            osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
            """.format(message, title, subtitle))


def marginal_ecdf(x, xp, up):
    x_sorted = np.sort(x.copy())
    xp_sorted = np.sort(xp.copy())
    up_sorted = np.sort(up.copy())
    u = np.interp(x_sorted, xp_sorted, up_sorted)
    return u


def ecdf(ds, ds_train, index='time'):
    """""
    Interpolate the empirical CDF of the training data onto the pretraining data.

    https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
    """
    X = ds_train['anomaly']
    U = ds_train['uniform']
    x = ds['anomaly']
    
    u = xr.apply_ufunc(
        marginal_ecdf, 
        x, X, U,
        input_core_dims=[[index], [index], [index]],
        output_core_dims=[[index]],
        exclude_dims={index},
        join='outer',
        vectorize=True
    )
    return u


def main(datadir):
    # load the data
    ds = xr.open_dataset(os.path.join(datadir, "data_1950_2022.nc"))
    u10 = ds.u10.values.flatten()

    if VISUALISATIONS:
        plt.hist(u10, bins=100)
        plt.yscale('log')

        # look at event duration histograms to select window length(s)
        ds_train = xr.open_dataset(os.path.join(datadir, "data.nc"))
        ds_train['duration'].plot.hist()
    
    # deseasonalise (with monthly medians)
    monthly = ds.groupby('time.month').median() 
    ds = ds.groupby('time.month') - monthly

    # make normal climate footprints       
    u10s = []
    tps = []
    msls = []
    times = []
    windows = []
    for window in WINDOWS:
        u10 = sliding_windows(ds['u10'].values, window)
        tp = sliding_windows(ds['tp'].values, window)
        msl = sliding_windows(ds['msl'].values, window)
        time = sliding_windows(ds['time'].values, window)
        window_length = [window] * u10.shape[0]

        # reduce arrays along window dimensions
        u10 = u10.max(axis=1)
        tp = tp.sum(axis=1)
        msl = msl.sum(axis=1)
        time = np.quantile(time, 0.5, axis=1) # middle time

        # append to list
        u10s.append(u10)
        tps.append(tp)
        msls.append(msl)
        times.append(time)
        windows.append(window_length)

    u10 = np.concatenate(u10s, axis=0)
    tp = np.concatenate(tps, axis=0)
    msl = np.concatenate(msls, axis=0)
    time = np.concatenate(times, axis=0)
    window_length = np.concatenate(windows, axis=0)
    X = np.stack([u10, tp, msl], axis=-1)

    #  make a new xarray dataset
    ds_window = xr.Dataset(
        {
            'anomaly': (('time', 'latitude', 'longitude', 'channel'), X),
            'window_length': (('time',), window_length)
        },
        coords={
            'time': time, #range(u10.shape[0]),
            'latitude': ds.latitude,
            'longitude': ds.longitude,
            'channel': ['u10', 'tp', 'mslp']
            }
    )
    ds_window = ds_window.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds_window

    # add metadata
    ds.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['script'] = f"scripts/make_pretraining.py"
    ds.attrs['last git commit'] = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode('UTF-8')
    ds.attrs['git branch'] = subprocess.Popen(["git", "branch", "--show-current"], stdout=subprocess.PIPE).communicate()[0].decode('UTF-8')
    ds.attrs['project'] = 'hazGAN'
    ds.attrs['note'] = "PIT by interpolation from storm data."

    # interpolate the empirical CDF
    ds['uniform'] = ecdf(ds, ds_train)

    if VISUALISATIONS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ds.isel(lon=0, lat=0, channel=0).anomaly.plot.hist(ax=axs[0, 0])
        ds.isel(lon=0, lat=0, channel=0).uniform.plot.hist(ax=axs[0, 1])
        ds.isel(time=100, channel=0).anomaly.plot.contourf(ax=axs[1, 0], levels=20, cmap='Spectral_r')
        ds.isel(time=100, channel=0).uniform.plot.contourf(ax=axs[1, 1], levels=20, cmap='Spectral_r')

    # Save to netCDF and NumPy
    ds.to_netcdf(os.path.join(datadir, "data_pretrain.nc"))
    np.savez(os.path.join(datadir, "data_pretrain.npz"), data=X)
    notify("Process finished", "Python script", "Finished making pretraining data")
    ds.close()


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")
    main(datadir)