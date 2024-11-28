"""
Process the GPD-fitted marginals to make training dataset for the GAN.

Input files:
------------
    - data_1950_2022.nc | source: scripts/data_processing/process_resampled.py
    - fitted_data.parquet | source: scripts/data_processing/marginals.R
    - event_data.parquet |  source: scripts/data_processing/marginals.R
    - monthly_medians.csv | source: scripts/data_processing/marginals.R
Output files:
-------------
    - data.nc
"""
# %%
import os
os.environ["USE_PYGEOS"] = "0"
from environs import Env
import subprocess
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from datetime import datetime
from calendar import month_name as month

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
from cartopy import crs as ccrs


plt.rcParams['font.family'] = 'serif'

CHANNELS = ["u10", "tp", 'mslp']
RESOLUTION = (22, 18)
VISUALISATIONS = False


def main(datadir):
    # load coordinates
    coords = xr.open_dataset(os.path.join(datadir, 'data_1950_2022.nc'))
    coords = coords['grid'].to_dataframe().reset_index()
    coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords['longitude'], coords['latitude'])).set_crs("EPSG:4326")

    # load GPD-fitted data                                                                                                        
    df = pd.read_parquet(os.path.join(datadir, f"fitted_data.parquet"))
    df = df.merge(coords, on="grid")
    df.columns = [col.replace(".", "_") for col in df.columns]
    df = df.rename(columns={"msl": "mslp"})
    df['day_of_storm'] = df.groupby('storm')['time_u10'].rank('dense')

    #  add event sizes
    events = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
    rate = events['lambda'][0]
    events = events[["storm", "storm.size", "lambda"]].groupby("storm").mean()
    events = events.to_dict()["storm.size"]
    df["size"] = df["storm"].map(events)
    gdf = gpd.GeoDataFrame(df, geometry="geometry").set_crs("EPSG:4326")

    # load event time data
    events = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
    times = pd.to_datetime(events[['storm', 'time']].groupby('storm').first()['time'].reset_index(drop=True))

    # Check fit quality and that it looks right
    if VISUALISATIONS:
        for var in  CHANNELS:
            p_crit = 0.1
            s0 = gdf["storm"].min()
            fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True,
            subplot_kw={'projection': ccrs.PlateCarree()})

            cmap = "PuBu_r"
            p_cmap = plt.get_cmap(cmap)
            p_cmap.set_under("crimson")

            gdf[gdf["storm"] == s0].plot(column=f"p_{var}", marker="s", cmap=p_cmap, vmin=p_crit, ax=axs[0])
            gdf[gdf["storm"] == s0].plot(column=f"thresh_{var}", legend=True, marker="s", cmap=cmap, ax=axs[1])
            gdf[gdf["storm"] == s0].plot(column=f"scale_{var}", legend=True, marker="s", cmap=cmap, ax=axs[2])
            gdf[gdf["storm"] == s0].plot(column=f"shape_{var}", legend=True, marker="s", cmap=cmap, ax=axs[3])

            # extend p-values colorbar to show where H0 rejected
            scatter = axs[0].collections[0]
            plt.colorbar(scatter, ax=axs[0], extend="min")

            axs[0].set_title("H₀: X~GPD(ξ,μ,σ)")
            axs[1].set_title("μ")
            axs[2].set_title("σ")
            axs[3].set_title("ξ")

            for ax in axs.ravel():
                ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

            fig.suptitle(f"Fit for ERA5 {var.upper()}, n = {gdf['storm'].nunique()}")
            print(gdf[gdf[f"p_{var}"] < p_crit]["grid"].nunique(), "significant p-values")

    # important: check ecdfs are in (0, 1)
    assert gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].max().max() <= 1, "ECDF values should be between 0 and 1"
    assert gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].min().min() >= 0, "ECDF values should be between 0 and 1"

    # merge in monthly medians
    monthly_medians = pd.read_csv(os.path.join(datadir, "monthly_medians.csv"), index_col="month")
    assert monthly_medians.groupby(['month', 'grid']).count().max().max() == 1, "Monthly medians not unique"
    monthly_medians = monthly_medians.groupby(["month", "grid"]).mean().reset_index()

    for var in CHANNELS:
        gdf[f"month_{var}"] = pd.to_datetime(gdf[f"time_{var}"]).dt.month.map(lambda x: month[x])
        n = len(gdf)
        gdf = gdf.join(monthly_medians[['month', 'grid', var]].set_index(["month", "grid"]), on=[f"month_{var}", "grid"], rsuffix="_median")
        assert n == len(gdf), "Merge failed"
        del gdf[f'month_{var}']

    # use latitude and longitude columns to label grid points in (i,j) format
    gdf["latitude"] = gdf["geometry"].apply(lambda x: x.y)
    gdf["longitude"] = gdf["geometry"].apply(lambda x: x.x)
    gdf = gdf.sort_values(["latitude", "longitude", "storm"], ascending=[True, True, True])

    #  make netcdf file
    nchannels = len(CHANNELS)
    T = gdf["storm"].nunique()
    nx = gdf["longitude"].nunique()
    ny = gdf["latitude"].nunique()

    # make training tensors
    gdf = gdf.sort_values(["storm", "latitude", "longitude"], ascending=[True, True, True]) # [T, i, j, channel]
    grid = gdf["grid"].unique().reshape([ny, nx])
    lat = gdf["latitude"].unique()
    lon = gdf["longitude"].unique()
    X = gdf[CHANNELS].values.reshape([T, ny, nx, nchannels])
    D = gdf[["day_of_storm"]].values.reshape([T, ny, nx])
    U = gdf[[f"ecdf_{c}" for c in CHANNELS]].values.reshape([T, ny, nx, nchannels])
    M = gdf[[f"{c}_median" for c in CHANNELS]].values.reshape([T, ny, nx, nchannels])
    z = gdf[["storm", "storm_rp"]].groupby("storm").mean().values.reshape(T)
    s = gdf[["storm", "size"]].groupby("storm").mean().values.reshape(T)

    lifetime_max_wind = np.max((X + M)[..., 0], axis=(1,2))
    lifetime_total_precip = np.sum((X + M)[..., 1], axis=(1,2))
    print("Max wind speed:", lifetime_max_wind.max())
    print("Total precipitation:", lifetime_total_precip.max())

    #  parameters for GPD
    if len(CHANNELS) > 1:
        gpd_params = ([f"thresh_{var}" for var in CHANNELS] + [f"scale_{var}" for var in CHANNELS] + [f"shape_{var}" for var in CHANNELS])
    else:
        gpd_params = ["thresh", "scale", "shape"]

    gdf_params = (gdf[[*gpd_params, "longitude", "latitude"]].groupby(["latitude", "longitude"]).mean().reset_index())
    thresh = np.array(gdf_params[[f"thresh_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    scale = np.array(gdf_params[[f"scale_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    shape = np.array(gdf_params[[f"shape_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    params = np.stack([shape, thresh, scale], axis=-2)

    # make an xarray dataset for training
    ds = xr.Dataset({'uniform': (['time', 'lat', 'lon', 'channel'], U),
                    'anomaly': (['time', 'lat', 'lon', 'channel'], X),
                    'medians': (['time', 'lat', 'lon', 'channel'], M),
                    'day_of_storm': (['time', 'lat', 'lon'], D),
                    'storm_rp': (['time'], z),
                    'duration': (['time'], s),
                    'params': (['lat', 'lon', 'param', 'channel'], params),
                    },
                    coords={'lat': (['lat'], lat),
                            'lon': (['lon'], lon),
                            'time': times,
                            'channel': CHANNELS,
                            'param': ['shape', 'loc', 'scale']
                    },
                    attrs={'CRS': 'EPSG:4326',
                            'u10': '10m Wind Speed [ms-1]',
                            'tp': 'Total Precipitation [m]',
                            'mslp': 'Mean Sea Level Pressure [wind spePa]',
                            'yearly_freq': rate})


    # extra information about the dataset
    ds.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['script'] = f"scripts/make_training.py"
    ds.attrs['last git commit'] = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode('UTF-8')
    ds.attrs['git branch'] = subprocess.Popen(["git", "branch", "--show-current"], stdout=subprocess.PIPE).communicate()[0].decode('UTF-8')
    ds.attrs['project'] = 'hazGAN'
    ds.attrs['note'] = "Fixed interpolation: [0, 1] --> (0, 1)."

    # save
    ds.to_netcdf(os.path.join(datadir, "data.nc"))
    print("Saved to", os.path.join(datadir, "data.nc"))

    if VISUALISATIONS:
        # day of storm
        cmap = mpl.cm.YlOrRd
        t = np.random.uniform(0, T, 1).astype(int)[0]

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        ds_t = ds.isel(time=t, channel=0).uniform #+ ds.isel(time=t).medians
        ds_t.plot(ax=ax)
        ax.set_title(f"Storm {t}")

        ax = axs[1]
        ds_t = ds.isel(time=t).day_of_storm.astype(int)
        vmin = ds_t.min().values
        vmax = ds_t.max().values

        bounds = np.arange(0.5, 11.5, 1).tolist()
        ticks = np.arange(0, 12, 1).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ds_t.plot(cmap=cmap, norm=norm, ax=ax, cbar_kwargs={'ticks': ticks})
        ax.set_title(f"Day of storm {t}")

        #  view netcdf file
        warnings.warn("Change ds_t to (-ds_t) if looking at MSLP")
        t = np.random.uniform(0, T, 1).astype(int)[0]
        ds_t = ds.isel(time=t)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        ds_t.isel(channel=0).anomaly.plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        ds_t.isel(channel=1).anomaly.plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle("Anomaly")

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        ds_t.isel(channel=0).medians.plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        ds_t.isel(channel=1).medians.plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle(f"Median")

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        (ds_t.isel(channel=0).anomaly + ds_t.isel(channel=0).medians).plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        (ds_t.isel(channel=1).anomaly - ds_t.isel(channel=1).medians).plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle('Anomaly + Median')

        #  check the highest wind speed is also the highest return period
        highest_wind = ds.anomaly.isel(channel=0).max(dim=['lat', 'lon']).values.max()
        highest_rp_wind = ds.anomaly.isel(channel=0, time=ds.storm_rp.argmax()).max(dim=['lat', 'lon']).values
        assert highest_wind == highest_rp_wind, "Highest wind speed doesn't correspond to highest return period"

        # have a look at the highest return period event
        ds_outlier = ds.isel(time=ds.storm_rp.argmax())
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        wind_footprint = ds_outlier.anomaly.isel(channel=0) + ds_outlier.medians.isel(channel=0)
        precip_footprint = (ds_outlier.anomaly.isel(channel=1) + ds_outlier.medians.isel(channel=1))
        wind_footprint.plot(cmap='Spectral_r', ax=axs[0])
        precip_footprint.plot(cmap='PuBu', ax=axs[1])

        fig, axs = plt.subplots(1, 2)
        for i, ax in enumerate(axs):
            hist_kws = {'bins': 50, 'color': 'lightgrey', 'edgecolor': 'k'}
            ax.hist(ds.isel(channel=i).anomaly.values.ravel(), **hist_kws);
        
    ds.close()


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")
    main(datadir)

# %%