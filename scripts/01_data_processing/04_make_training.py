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
from pathlib import Path

from datetime import datetime
from calendar import month_name as month

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'figure.titlesize': 8,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'normal',
    'legend.fontsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'font.family': 'sans-serif'
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


def save_stats_text(path, stats_dict):
    with open(path, "w") as f:
        f.write(f"RESULTS SUMMARY\n{'='*20}\n\n")
        for section, values in stats_dict.items():
            f.write(f"[{section}]\n")
            for k, v in values.items():
                val_str = f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)
                f.write(f"{k:.<20} {val_str}\n")
            f.write("\n")


if __name__ == "__main__":

    # configure paths
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("PROCESSING_DIR")
    outdir = env.str("TRAINDIR")
    figdir = Path(env.str("FIG_DIR")) / "params"
    figdir.mkdir(parents=True, exist_ok=True)

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
    
    # %% Check fit quality and that it looks right
    paramplot = True
    if paramplot:

        alpha = 0.05
        ds = gdf.set_index(['lat', 'lon', 'storm']).to_xarray().isel(storm=0) # all params the same over storms
        
        results = {}

        for var in  FIELDS:

            fig = plt.figure(figsize=(6, 1.75), constrained_layout=True)
            gs = gridspec.GridSpec(
                2, 5, figure=fig,
                height_ratios=[1, 0.04],
                width_ratios=[1, 1, 1, 1, 1.4],
                wspace=0.0
            )

            # top row: plots
            axs = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(4)]
            ax4 = fig.add_subplot(gs[0, 4])

            # bottom row: colorbars
            caxs = [fig.add_subplot(gs[1, i]) for i in range(5)]

            p_cmap = plt.get_cmap(CMAP)
            p_cmap.set_under("crimson")

            im0 = ds[f"pk_{var}"].plot(ax=axs[0], cmap=p_cmap, vmin=alpha, vmax=1, add_colorbar=False)
            im1 = ds[f"thresh_{var}"].plot(ax=axs[1], cmap=CMAP, add_colorbar=False)
            im2 = ds[f"scale_{var}"].plot(ax=axs[2], cmap=CMAP, add_colorbar=False)
            im3 = ds[f"shape_{var}"].plot(ax=axs[3], cmap=CMAP, add_colorbar=False)
            
            # plot the density for three sample grid points
            dist = getattr(scipy.stats, 'genpareto')
            shapes_all = gdf[f'shape_{var}'].values
            percs = np.linspace(0.01, 0.99, 10)
            shapes = gdf[f'shape_{var}'].quantile(percs)
            scale  = gdf[f'scale_{var}'].mean()

            vmin = min(shapes_all)
            vmax = max(shapes_all)
            norm = plt.Normalize(vmin, vmax)
            colors = [plt.get_cmap(CMAP)(norm(value)) for value in shapes]
            
            for i, shape in enumerate(shapes):
                def fraction_formatter(x, pos):
                    return f'{x:.2f}'
                
                u = np.linspace(0.95, 0.999, 200)

                x = dist.ppf(u, shape)
                y = dist.pdf(x, shape)

                ax4.plot(x, y, linewidth=1, label=f"ξ={shape:.2f}", color=colors[i])
                ax4.set_xlabel("")
                ax4.set_ylabel("")
                ax4.yaxis.set_major_formatter(fraction_formatter)
                ax4.tick_params(direction='in')
                ax4.yaxis.set_label_position("right")
                ax4.set_box_aspect(0.7)
                ax4.spines['top'].set_visible(False)
                ax4.spines['right'].set_visible(False)
            
            cbar_kwargs = { # add colorbars
                'orientation': 'horizontal', 'shrink': 0.8,
                'aspect': 0.01, 'pad': 0.0
            }
            labels = ['p', 'μ', 'σ', 'ξ', 'ξ']
            extend = ['min', 'neither', 'neither', 'neither', 'neither']
            for i, im in enumerate([im0, im1, im2, im3, im3]):
                fig.colorbar(
                    im, cax=caxs[i], extend=extend[i], label=labels[i],
                    **cbar_kwargs
                )
                if i == 0: # tidy p-value colorbar ticks
                    caxs[i].set_xticks([0.25, 0.5, 0.75])

            for ax in axs:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.25)
                ax.set_title("")
            ax4.set_title("")

            results[var] = {}

            p_significant = gdf[gdf[f"pk_{var}"] < alpha]
            results[var]["significant p-values"] = p_significant["grid"].nunique()

            outpath = figdir / f"{var}.png"
            fig.savefig(outpath, dpi=300, transparent=True)
            print(f"Saved {var} parameters plot to {outpath}")

    #  important: assert ecdfs are in [0, 1)
    maxima = gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].max()
    minima = gdf[['ecdf_u10', 'ecdf_tp', 'ecdf_mslp']].min()

    assert maxima.max() < 1, "ECDF values should be between 0 and 1"
    assert minima.min() >= 0, "ECDF values should be between 0 and 1"

    for max_i, min_i, var in zip(maxima, minima, FIELDS):
        results[var]["max(ecdf)"] = f"{max_i:.6f}"
        results[var]["min(ecdf)"] = f"{min_i:.6f}"

        rpmax = 1 / (1 - max_i)
        rpmin = 1 / (1 - min_i)

        results[var]["max return period"] = f"{rpmax:,.0f}-yr"
        results[var]["min return period"] = f"{rpmin:,.0f}-yr"

    statspath = figdir / "summary.txt"
    save_stats_text(statspath, results)
    print(f"Saved summary statistics to {statspath}")

    # %% merge in monthly medians
    month_map = {name: num for num, name in enumerate(month) if num > 0}

    monthly_medians = pd.read_csv(os.path.join(datadir, INFILES[3]), index_col="month")
    assert monthly_medians.groupby(['month', 'grid']).count().max().max() == 1, "Monthly medians not unique"
    monthly_medians = monthly_medians.groupby(["month", "grid"]).mean().reset_index()
    monthly_medians['month'] = monthly_medians['month'].map(month_map)

    # assign lat, lon by grid index
    len_before = len(monthly_medians)
    monthly_medians = monthly_medians.merge(coords[['grid', 'lat', 'lon']], on='grid')
    assert len_before == len(monthly_medians), "Merge failed"
    monthly_medians = monthly_medians.sort_values(["month", "lat", "lon"], ascending=[True, True, True])
    del len_before, monthly_medians['grid']

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
    z    = gdf[["storm", "storm_rp"]].groupby("storm").mean().values.reshape(T)
    s    = gdf[["storm", "size"]].groupby("storm").mean().values.reshape(T)

    # monthly medians has a different shape
    M = monthly_medians[[f for f in FIELDS]].values.reshape([12, ny, nx, nfields])

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
                    'medians': (['month', 'lat', 'lon', 'field'], M),
                    'day_of_storm': (['time', 'lat', 'lon'], D),
                    'storm_rp': (['time'], z),
                    'duration': (['time'], s),
                    'params': (['lat', 'lon', 'param', 'field'], params),
                    'grid': (['lat', 'lon'], grid),
                    },
                    coords={'lat': (['lat'], lat),
                            'lon': (['lon'], lon),
                            'time': times,
                            'month': monthly_medians['month'].unique().reshape(12),
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

    # %% save
    print("Finished! Saving to netcdf...")
    ds.to_netcdf(os.path.join(outdir, OUTFILES[0]))
    print("Saved to", os.path.join(outdir, OUTFILES[0]))
        
    ds.close()
    print(f"U1 max in memory: {U1.max()}")

# %%