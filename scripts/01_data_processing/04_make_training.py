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
    - plots of marginal fit quality (if VISUALISATIONS=True)
"""
# %%
import os
os.environ["USE_PYGEOS"] = "0"
from environs import Env

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import scipy.stats

from datetime import datetime
from calendar import month_name as month

import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


FIELDS = ["u10", "tp", 'mslp']
VISUALISATIONS = True
PROCESS_OUTLIERS = False
THRESHOLD = 0.75 # (for outliers, not using)
RES = (64, 64)

# for snakemake in future
CMAP = "PuBu_r"
INFILES = ['data_1941_2022.nc', 'storms.parquet', 'storms_metadata.parquet', 'medians.csv']
OUTFILES = ['data.nc']

hist_kws = {
    'bins': 50,
    'color': 'lightgrey',
    'edgecolor': 'k',
    'density': True
}


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("PROCESSING_DIR")
    outdir = env.str("TRAINDIR")

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

    # add event sizes
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
            alpha = 0.05
            ds = gdf.set_index(['lat', 'lon', 'storm']).to_xarray().isel(storm=0)


            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(7, 2), constrained_layout=True)
            gs = gridspec.GridSpec(2, 5, height_ratios=[1, 0.04], width_ratios=[1, 1, 1, 1, 1.4],
                                wspace=0.0, figure=fig
            )

            # Top row: plots
            axs = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(4)]
            ax4 = fig.add_subplot(gs[0, 4])

            # Bottom row: colorbars
            caxs = [fig.add_subplot(gs[1, i]) for i in range(5)]

            p_cmap = plt.get_cmap(CMAP)
            p_cmap.set_under("crimson")

            im0 = ds[f"pk_{var}"].plot(ax=axs[0], cmap=p_cmap, vmin=alpha, add_colorbar=False)
            im1 = ds[f"thresh_{var}"].plot(ax=axs[1], cmap=CMAP, add_colorbar=False)
            im2 = ds[f"scale_{var}"].plot(ax=axs[2], cmap=CMAP, add_colorbar=False)
            im3 = ds[f"shape_{var}"].plot(ax=axs[3], cmap=CMAP, add_colorbar=False)

            # add colorbars manually for better control
            #! increased shrink and halved aspect 12 -> 6
            #! changed aspect back to 12
            #! changes aspect to 24
            #! changed aspect to 1000
            #! changed shrink to 0.01
            cbar_kwargs = {'orientation': 'horizontal', 'shrink': 1.0, 'aspect': 0.01,
                           'pad': 0.0}
            labels = ['p', 'μ', 'σ', 'ξ', 'ξ']
            for cax, im, label in zip(caxs, [im0, im1, im2, im3, im3], labels):
                fig.colorbar(im, cax=cax, **cbar_kwargs).set_label(label)
            
            # plot the density for three sample grid points
            dist = getattr(scipy.stats, 'genpareto')
            shapes_all = gdf[f'shape_{var}'].values
            percentiles = np.linspace(0.01, 0.99, 10)
            shapes = gdf[f'shape_{var}'].quantile(percentiles)
            scale  = gdf[f'scale_{var}'].mean()

            vmin = min(shapes_all)
            vmax = max(shapes_all)
            norm = plt.Normalize(vmin, vmax)
            colors = [plt.get_cmap(CMAP)(norm(value)) for value in shapes]
            
            for i, shape in enumerate(shapes):
                u = np.linspace(0.95, 0.999, 100)
                x = dist.ppf(u, shape)
                y = dist.pdf(x, shape)
                ax4.plot(x, y, linewidth=1, label=f"ξ={shape:.2f}", color=colors[i])
                
                def fraction_formatter(x, pos):
                    return f'{x:.2f}'
                
                ax4.set_xlabel("")
                ax4.set_ylabel("")
                ax4.yaxis.set_major_formatter(fraction_formatter)
                ax4.tick_params(direction='in')
                ax4.yaxis.set_label_position("right")
                ax4.set_box_aspect(0.7)
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)

            axs[0].set_title("")
            axs[1].set_title("")
            axs[2].set_title("")
            axs[3].set_title("")
            ax4.set_title("")

            for ax in axs[:-1]:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

            print(gdf[gdf[f"pk_{var}"] < alpha]["grid"].nunique(), "significant p-values")

    #  important: check ecdfs are in [0, 1)
    maxima = gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].max()
    minima = gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].min()
    print(f"{maxima=}")
    print(f"{minima=}")

    assert maxima.max() < 1, "ECDF values should be between 0 and 1"
    assert minima.min() >= 0, "ECDF values should be between 0 and 1"

    for max_i, min_i, var in zip(maxima, minima, FIELDS):
        print(f"ECDF {var}: min {min_i:.6f}, max {max_i:.6f}")
        print(f"Corresponds to {1/(1 - max_i):,.0f}-year max return level")
        print(f"Corresponds to {1/(1 - min_i):,.0f}-year min return level")


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
    ds = xr.Dataset({'uniform': (['time', 'lat', 'lon', 'field'], U0),
                    'ecdf': (['time', 'lat', 'lon', 'field'], U0),
                    'scdf': (['time', 'lat', 'lon', 'field'], U1),
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
    ds.attrs['project'] = 'hazGAN'
    ds.attrs['note'] = "Fixed interpolation: [0, 1] --> (0, 1)."

    # save
    print("Finished! Saving to netcdf...")
    ds.to_netcdf(os.path.join(outdir, OUTFILES[0]))
    print("Saved to", os.path.join(outdir, OUTFILES[0]))
        
    ds.close()
    print(f"U1 max in memory: {U1.max()}")

# %%