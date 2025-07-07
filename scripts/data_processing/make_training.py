"""
Process the GPD-fitted marginals to make training dataset for the GAN.

Input files:
------------
    - data_1940_2022.nc        | source: scripts/data_processing/process_resampled.py
    - storms.parquet           | source: scripts/data_processing/marginals.R
    - storms_metadata.parquet  | source: scripts/data_processing/marginals.R
    - medians.csv              | source: scripts/data_processing/marginals.R

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

from hazGAN.utils import res2str
from hazGAN.plotting import palettes


plt.rcParams['font.family'] = 'serif'
hist_kws = {'bins': 50, 'color': 'lightgrey', 'edgecolor': 'k', 'density': True}

FIELDS = ["u10", "tp", 'mslp']
VISUALISATIONS = True
SAVE_DATA = False
THRESHOLD = 0.75 # (for outliers, not using)
PROCESS_OUTLIERS = False
RES = (64, 64)

# for snakemake in future
INFILES = ['data_1941_2022.nc', 'storms.parquet', 'storms_metadata.parquet', 'medians.csv']
OUTFILES = ['data.nc']


def main(datadir):
    # load coordinates
    coords = xr.open_dataset(os.path.join(datadir, INFILES[0]))
    coords = coords['grid'].to_dataframe().reset_index()
    coords = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords['lon'], coords['lat'])
        ).set_crs("EPSG:4326")

    # load GPD-fitted data                                                                                                        
    df = pd.read_parquet(os.path.join(datadir, INFILES[1]))

    df = df.merge(coords, on="grid")
    df.columns = [col.replace(".", "_") for col in df.columns]
    df = df.rename(columns={"msl": "mslp"})
    df['day_of_storm'] = df.groupby('storm')['time_u10'].rank('dense')

    #  add event sizes
    events = pd.read_parquet(os.path.join(datadir, INFILES[2]))
    rate = events['lambda'][0]
    events = events[["storm", "storm.size", "lambda"]].groupby("storm").mean()
    events = events.to_dict()["storm.size"]
    df["size"] = df["storm"].map(events)
    gdf = gpd.GeoDataFrame(df, geometry="geometry").set_crs("EPSG:4326")

    # load event time data
    events = pd.read_parquet(os.path.join(datadir, INFILES[2]))
    times = pd.to_datetime(events[['storm', 'time']].groupby('storm').first()['time'].reset_index(drop=True))
    
    # Check fit quality and that it looks right
    if VISUALISATIONS:
        for var in  FIELDS:
            p_crit = 0.05
            # s0 = gdf["storm"].min()

            ds = gdf.set_index(['lat', 'lon', 'storm']).to_xarray().isel(storm=0)

            fig, axs = plt.subplots(1, 4, figsize=(16, 3), sharex=True, sharey=True,
            subplot_kw={'projection': ccrs.PlateCarree()})

            # ax4 = fig.add_subplot(1, 5, 5, projection=None)
            ax4 = fig.add_axes([0.825, 0.1, 0.15, 0.8])   # [left, bottom, width, height]
            plt.tight_layout()
            plt.subplots_adjust(right=0.8) 

            cmap = "PuBu_r" # "viridis" # "PuBu_r"
            p_cmap = plt.get_cmap(cmap)
            p_cmap.set_under("crimson")

            ds[f"pk_{var}"].plot(ax=axs[0], cmap=p_cmap, vmin=p_crit, cbar_kwargs={'label': None})
            ds[f"thresh_{var}"].plot(ax=axs[1], cmap=cmap, cbar_kwargs={'label': None})
            ds[f"scale_{var}"].plot(ax=axs[2], cmap=cmap, cbar_kwargs={'label': None})
            ds[f"shape_{var}"].plot(ax=axs[3], cmap=cmap, add_colorbar=False) #, vmin=-0.81, vmax=0.28)

            # plot the density for three sample grid points
            import scipy.stats
            if var == 'u10':
                dist = getattr(scipy.stats, 'weibull_min')
            else:
                dist = getattr(scipy.stats, 'genpareto')

            # plot some densities
            shapes_all = gdf[f'shape_{var}'].values
            percentiles = np.linspace(0.01, 0.99, 10)
            shapes = gdf[f'shape_{var}'].quantile(percentiles)
            loc    = gdf[f'thresh_{var}'].mean()
            scale  = gdf[f'scale_{var}'].mean()

            vmin = min(shapes_all)
            vmax = max(shapes_all)
            norm = plt.Normalize(vmin, vmax)
            colors = [plt.get_cmap(cmap)(norm(value)) for value in shapes]
            
            for i, shape in enumerate(shapes):
                u = np.linspace(0.95, 0.999, 100)
                x = dist.ppf(u, shape) #, loc=loc, scale=scale)
                y = dist.pdf(x, shape) #, loc=loc, scale=scale)
                ax4.plot(x, y, label=f"ξ={shape:.2f}", color=colors[i])

                # axis cleanup
                def percentage_formatter(x, pos):
                    return f'{100 * x:.0f}%'  # Multiply by 100 and add % sign
                
                ax4.set_xlabel("")
                ax4.set_ylabel("")
                ax4.yaxis.set_major_formatter(percentage_formatter)
                ax4.tick_params(direction='in')
                ax4.yaxis.set_label_position("right")
                # ax4.tick_params(axis='y', pad=-40)

            # add colorbar for shape
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax4)
            # set label to none
            cbar.set_label(None)
            cbar.set_label("ξ")
            # put cbar label on LEFT of it
            cbar.ax.yaxis.set_label_position('left')
            # rotate the cbar label to be UPRIGHT
            cbar.ax.set_ylabel('ξ', rotation=0, labelpad=15)
            # cbar.set_ticks(shapes)
            # cbar.set_ticklabels([f"{s:.2f}" for s in shapes])

            # ax4.legend()

            if var == 'u10':
                axs[0].set_title("H₀: X~Weibull(ξ,μ,σ)")
            else:
                axs[0].set_title("H₀: X~GPD(ξ,μ,σ)")

            axs[1].set_title("μ")
            axs[2].set_title("σ")
            axs[3].set_title("ξ")

            for ax in axs[:-1].ravel():
                ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

            # fig.suptitle(f"Fit for ERA5 {var.upper()}, n = {gdf['storm'].nunique()}")
            print(gdf[gdf[f"pk_{var}"] < p_crit]["grid"].nunique(), "significant p-values")

            if False:
                fig, axs = plt.subplots(1, 4, figsize=(18, 3))
                gdf['pk_u10'].hist(ax=axs[0], **hist_kws)
                gdf[f"thresh_{var}"].hist(ax=axs[1], **hist_kws)
                gdf[f"scale_{var}"].hist(ax=axs[2], **hist_kws)
                gdf[f"shape_{var}"].hist(ax=axs[3], **hist_kws)
                axs[1].set_title("μ")
                axs[2].set_title("σ")
                axs[3].set_title("ξ")
                
            fig.savefig(f"/Users/alison/Desktop/f01_{var}.png", dpi=300)

    # return gdf # TODO: remove this line later
    #  important: check ecdfs are in (0, 1)
    assert gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].max().max() <= 1, "ECDF values should be between 0 and 1"
    assert gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].min().min() >= 0, "ECDF values should be between 0 and 1"

    # merge in monthly medians
    monthly_medians = pd.read_csv(os.path.join(datadir, INFILES[3]), index_col="month")
    assert monthly_medians.groupby(['month', 'grid']).count().max().max() == 1, "Monthly medians not unique"
    monthly_medians = monthly_medians.groupby(["month", "grid"]).mean().reset_index()

    for var in FIELDS:
        gdf[f"month_{var}"] = pd.to_datetime(gdf[f"time_{var}"]).dt.month.map(lambda x: month[x])
        n = len(gdf)
        gdf = gdf.join(monthly_medians[['month', 'grid', var]].set_index(["month", "grid"]), on=[f"month_{var}", "grid"], rsuffix="_median")
        assert n == len(gdf), "Merge failed"
        del gdf[f'month_{var}']

    # use lat and lon columns to label grid points in (i,j) format
    gdf["lat"] = gdf["geometry"].apply(lambda x: x.y)
    gdf["lon"] = gdf["geometry"].apply(lambda x: x.x)
    gdf = gdf.sort_values(["lat", "lon", "storm"], ascending=[True, True, True])

    #  make netcdf file
    nfields = len(FIELDS)
    nx      = gdf["lon"].nunique()
    ny      = gdf["lat"].nunique()
    T       = gdf["storm"].nunique()

    # make training tensors
    gdf  = gdf.sort_values(["storm", "lat", "lon"], ascending=[True, True, True]) # [T, i, j, field]
    grid = gdf["grid"].unique().reshape([ny, nx])
    lat  = gdf["lat"].unique()
    lon  = gdf["lon"].unique()
    X    = gdf[FIELDS].values.reshape([T, ny, nx, nfields])
    D    = gdf[["day_of_storm"]].values.reshape([T, ny, nx])
    U0   = gdf[[f"ecdf_{c}" for c in FIELDS]].values.reshape([T, ny, nx, nfields])
    U1   = gdf[[f"scdf_{c}" for c in FIELDS]].values.reshape([T, ny, nx, nfields])
    M    = gdf[[f"{c}_median" for c in FIELDS]].values.reshape([T, ny, nx, nfields])
    z    = gdf[["storm", "storm_rp"]].groupby("storm").mean().values.reshape(T)
    s    = gdf[["storm", "size"]].groupby("storm").mean().values.reshape(T)

    lifetime_max_wind = np.max((X + M)[..., 0], axis=(1,2))
    lifetime_total_precip = np.sum((X + M)[..., 1], axis=(1,2))
    print("Max wind speed:", lifetime_max_wind.max())
    print("Total precipitation:", lifetime_total_precip.max())

    #  parameters for GPD
    if len(FIELDS) > 1:
        gpd_params = ([f"thresh_{var}" for var in FIELDS] + [f"scale_{var}" for var in FIELDS] + [f"shape_{var}" for var in FIELDS])
    else:
        gpd_params = ["thresh", "scale", "shape"]

    gdf_params = (gdf[[*gpd_params, "lon", "lat"]].groupby(["lat", "lon"]).mean().reset_index())
    thresh = np.array(gdf_params[[f"thresh_{var}" for var in FIELDS]].values.reshape([ny, nx, nfields]))
    scale = np.array(gdf_params[[f"scale_{var}" for var in FIELDS]].values.reshape([ny, nx, nfields]))
    shape = np.array(gdf_params[[f"shape_{var}" for var in FIELDS]].values.reshape([ny, nx, nfields]))
    params = np.stack([thresh, scale, shape], axis=-2)

    # make an xarray dataset for training
    # NOTE: using SemiCDF instead of ECDF because invPIT works better
    ds = xr.Dataset({'uniform': (['time', 'lat', 'lon', 'field'], U1),
                    'ecdf': (['time', 'lat', 'lon', 'field'], U0),
                    'anomaly': (['time', 'lat', 'lon', 'field'], X),
                    'medians': (['time', 'lat', 'lon', 'field'], M),
                    'day_of_storm': (['time', 'lat', 'lon'], D),
                    'storm_rp': (['time'], z),
                    'duration': (['time'], s),
                    'params': (['lat', 'lon', 'param', 'field'], params),
                    'grid': (['lat', 'lon'], grid),
                    },
                    coords={'lat': (['lat'], lat),
                            'lon': (['lon'], lon),
                            'time': times,
                            'field': FIELDS,
                            'param': ['loc', 'scale', 'shape']
                    },
                    attrs={'CRS': 'EPSG:4326',
                            'u10': '10m Wind Speed [ms-1]',
                            'tp': 'Total Precipitation [m]',
                            'mslp': 'Mean Sea Level Pressure [Pa]',
                            'yearly_freq': rate})


    # extra information about the dataset
    ds.attrs['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['script'] = f"scripts/make_training.py"
    ds.attrs['last git commit'] = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode('UTF-8')
    ds.attrs['git branch'] = subprocess.Popen(["git", "branch", "--show-current"], stdout=subprocess.PIPE).communicate()[0].decode('UTF-8')
    ds.attrs['project'] = 'hazGAN'
    ds.attrs['note'] = "Fixed interpolation: [0, 1] --> (0, 1)."

    # remove outliers
    # if PROCESS_OUTLIERS:
        # ds = process_outliers(ds, THRESHOLD, datadir=datadir, visuals=VISUALISATIONS)

    # save
    if SAVE_DATA:
        print("Finished! Saving to netcdf...")
        # ds.to_netcdf(os.path.join(datadir, OUTFILES[0]))
        print("Saved to", os.path.join(datadir, OUTFILES[0]))

    if False:
        # day of storm
        cmap = mpl.cm.YlOrRd
        t = np.random.uniform(0, T, 1).astype(int)[0]

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        ax = axs[0]
        ds_t = ds.isel(time=t, field=0).uniform #+ ds.isel(time=t).medians
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

        # view netcdf file
        warnings.warn("Change ds_t to (-ds_t) if looking at MSLP")
        t = np.random.uniform(0, T, 1).astype(int)[0]
        ds_t = ds.isel(time=t)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        ds_t.isel(field=0).anomaly.plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        ds_t.isel(field=1).anomaly.plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle("Anomaly")

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        ds_t.isel(field=0).medians.plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        ds_t.isel(field=1).medians.plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle(f"Median")

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        (ds_t.isel(field=0).anomaly + ds_t.isel(field=0).medians).plot.contourf(cmap='Spectral_r', ax=axs[0], levels=20)
        (ds_t.isel(field=1).anomaly - ds_t.isel(field=1).medians).plot.contourf(cmap='PuBu', ax=axs[1], levels=15)
        fig.suptitle('Anomaly + Median')

        # check the highest wind speed is also the highest return period
        highest_wind = ds.anomaly.isel(field=0).max(dim=['lat', 'lon']).values.max()
        highest_rp_wind = ds.anomaly.isel(field=0, time=ds.storm_rp.argmax()).max(dim=['lat', 'lon']).values
        assert highest_wind == highest_rp_wind, "Highest wind speed doesn't correspond to highest return period"

        # have a look at the highest return period event
        ds_outlier = ds.isel(time=ds.storm_rp.argmax())
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        wind_footprint = ds_outlier.anomaly.isel(field=0) + ds_outlier.medians.isel(field=0)
        precip_footprint = (ds_outlier.anomaly.isel(field=1) + ds_outlier.medians.isel(field=1))
        wind_footprint.plot(cmap='Spectral_r', ax=axs[0])
        precip_footprint.plot(cmap='PuBu', ax=axs[1])

        fig, axs = plt.subplots(1, 2, figsize=(16, 4))
        for i, ax in enumerate(axs):
            ax.hist(ds.isel(field=i).anomaly.values.ravel(), **hist_kws);
        
    ds.close()
    return gdf

# %%
if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = os.path.join(env.str("TRAINDIR"), res2str(RES))
    gdf = main(datadir)

# %%