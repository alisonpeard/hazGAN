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


plt.rcParams['font.family'] = 'serif'
hist_kws = {'bins': 50, 'color': 'lightgrey', 'edgecolor': 'k'}

CHANNELS = ["u10", "tp", 'mslp']
RESOLUTION = (22, 18)
VISUALISATIONS = True
THRESHOLD = 0.75 # rough manual bisection for this
PROCESS_OUTLIERS = True

# for snakemake in future
INFILES = ['data_1940_2022.nc', 'storms.parquet', 'storms_metadata.parquet', 'medians.csv']
OUTFILES = ['data.nc']


def frobenius(test:np.ndarray, template:np.ndarray) -> float:
    """Calculate the Frobenius norm (similarity) of two matrices."""
    similarity = np.sum(template * test) / (np.linalg.norm(template) * np.linalg.norm(test))
    return similarity


def process_outliers(ds:xr.Dataset, threshold:float=THRESHOLD,
                     datadir:str='.', visuals:bool=VISUALISATIONS) -> xr.Dataset:
    """Remove "wind bomb" outliers from data
    
    Args:
    -----
    ds: xr.Dataset
        The dataset to process
    threshold: float
        The threshold for the Frobenius norm similarity between the template and the test matrix.
    """
    ds = ds.copy()
    ds['maxwind'] = ds.sel(channel='u10')['anomaly'].max(dim=['lon', 'lat'])

    sorting = ds.sel(channel='u10')['maxwind'].argsort()

    if visuals:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ds['anomaly'].isel(channel=0, time=sorting[-1]).plot(ax=axs[0])
        axs[0].set_title('Largest sample before filtering.')

    if not os.path.exists(os.path.join(datadir, "windbomb.npy")):
        print('Creating outlier template.')
        template = ds['anomaly'].isel(channel=0, time=sorting[-1]).data # this has a "wind bomb"
        np.save(os.path.join(datadir, "windbomb.npy"), template)    
    else:
        print('Loading outlier template.')
        template = np.load(os.path.join(datadir, "windbomb.npy"))

    similarities = [] * ds['time'].data.size
    for i in range(ds['time'].data.size):
        test_matrix = ds['anomaly'].isel(channel=0, time=i).data
        similarity = frobenius(test_matrix, template)
        similarities.append(similarity)
    similarities = np.array(similarities)

    print(f'{sum(similarities > threshold)} ERA5 "wind bombs" detected in dataset for threshold {threshold}.')
    mask = similarities <= threshold 

    ds_filtered = ds.sel(time=mask)
    sorting = ds_filtered.sel(channel='u10')['maxwind'].argsort().data
    ds_filtered = ds_filtered.isel(time=sorting[-60000:]) # 60,000 like MNIST
    sorting = ds_filtered.sel(channel='u10')['maxwind'].argsort().data # again

    if visuals:
        ds_filtered['anomaly'].isel(channel=0, time=sorting[-1]).plot(ax=axs[1])
        axs[1].set_title('Largest sample after filtering.')

    print("Returning 60,000 largest filtered samples.")
    return ds_filtered


def main(datadir):
    # load coordinates
    coords = xr.open_dataset(os.path.join(datadir, INFILES[0]))
    coords = coords['grid'].to_dataframe().reset_index()
    coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords['lon'], coords['lat'])).set_crs("EPSG:4326")

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
    monthly_medians = pd.read_csv(os.path.join(datadir, INFILES[3]), index_col="month")
    assert monthly_medians.groupby(['month', 'grid']).count().max().max() == 1, "Monthly medians not unique"
    monthly_medians = monthly_medians.groupby(["month", "grid"]).mean().reset_index()

    for var in CHANNELS:
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
    nchannels = len(CHANNELS)
    T = gdf["storm"].nunique()
    nx = gdf["lon"].nunique()
    ny = gdf["lat"].nunique()

    # make training tensors
    gdf = gdf.sort_values(["storm", "lat", "lon"], ascending=[True, True, True]) # [T, i, j, channel]
    grid = gdf["grid"].unique().reshape([ny, nx])
    lat = gdf["lat"].unique()
    lon = gdf["lon"].unique()
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

    gdf_params = (gdf[[*gpd_params, "lon", "lat"]].groupby(["lat", "lon"]).mean().reset_index())
    thresh = np.array(gdf_params[[f"thresh_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    scale = np.array(gdf_params[[f"scale_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    shape = np.array(gdf_params[[f"shape_{var}" for var in CHANNELS]].values.reshape([ny, nx, nchannels]))
    params = np.stack([thresh, scale, shape], axis=-2)

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
                            'param': ['loc', 'scale', 'shape']
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

    # remove outliers
    # ds = process_outliers(ds, THRESHOLD, datadir=datadir, visuals=VISUALISATIONS)

    # save
    print("Finished! Saving to netcdf...")
    ds.to_netcdf(os.path.join(datadir, OUTFILES[0]))
    print("Saved to", os.path.join(datadir, OUTFILES[0]))

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
            ax.hist(ds.isel(channel=i).anomaly.values.ravel(), **hist_kws);
        
    ds.close()


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")
    main(datadir)

# %% MAKING TESTS
# to add to pytest later
# data_1940_2022 = xr.open_dataset(os.path.join(datadir, 'data_1940_2022.nc'))
metadata = pd.read_parquet(os.path.join(datadir, "storms_metadata.parquet"))
storms   = pd.read_parquet(os.path.join(datadir, "storms.parquet"))
data     = xr.open_dataset(os.path.join(datadir, OUTFILES[0]))

# %%
# TEST 1: does data_1940_2022.nc match metadata – YES
# TEST 2: does storms_metadata match storms.parquet - ?
# TEST 3: does storms.parquet match data.nc – ?

# %% TEST 2
a = metadata[['time', 'u10']].copy()
a['time'] = pd.to_datetime(a['time'])
a = a.set_index('time', drop=True)
a = a.sort_index()

b = storms[['time.u10', 'u10']].copy()
b.columns = ['time', 'u10']
b['time'] = pd.to_datetime(b['time'])
b = b.groupby('time').max()

c = b.join(a, how='left', lsuffix='_b', rsuffix='_a')
c['difference'] = c['u10_a'] - c['u10_b']
assert c['difference'].sum() == 0

# %%


# %% THIS MORNING'S STUFF
# %% ecdf of wind maxima
from scipy.stats import rankdata
data = xr.open_dataset(os.path.join(datadir, OUTFILES[0]))
data = data.sel(time=slice('1940', '2021'))

# %%
for i in range(data.dims['channel']):
    anomaly = data.isel(channel=i)['anomaly'].values
    uniform = data.isel(channel=i)['uniform'].values

    n, h, w = anomaly.shape
    anomaly = anomaly.reshape(n, h*w)
    uniform = uniform.reshape(n, h*w)

    for gridcell in range(h*w):
        x = sorted(anomaly[:, gridcell])
        u = sorted(uniform[:, gridcell])
        
        x = rankdata(x) / (n + 1)
        assert np.allclose(x, u, atol=1e-3), f"ECDFs for gridcell {gridcell} are not equal."

    # n = len(anomaly)
    # anomaly = so
# %% test data.nc anomaly+medians matches original
data = xr.open_dataset(os.path.join(datadir, 'data.nc'))
original = xr.open_dataset(os.path.join(datadir, 'data_1940_2022.nc'))

u10_train = (data['anomaly'] + data['medians']).isel(channel='u10')

# %%
original = original['maxwind'].to_dataframe('u10')

# %% dev
pd.set_option('display.max_colwidth', None)
before = metadata.groupby('storm').apply(lambda x: pd.Series({
    'u10': x['u10'].max(),
    'time': x['time'][x['u10'].idxmax()],
    'timelist': str(list(x['time'])),
    'u10list': str(list(x['u10']))
}))

before['time'] = pd.to_datetime(before['time'])
before = before.set_index('time', drop=True)

u10 = data.sel(channel="u10")['anomaly']
after = u10.max(dim=['lat', 'lon'])
after = after.to_dataframe('u10').drop(columns=['channel'])
after = after.sort_values('time')

comparison = pd.concat([before, after], axis=1)
comparison.iloc[[3],:]
# %%
# look at original data 1940-02-26 - 1940-03-27
original = xr.open_dataset(os.path.join(datadir, 'data_1940_2022.nc'))
original = original['maxwind'].to_dataframe('u10')
original = original.reset_index()
original['time'] = pd.to_datetime(original['time'])
original = original.sort_values('time')
original = original.set_index('time', drop=True)
original.loc['1940-02-26':'1940-03-27']
# %%

start = pd.to_datetime('1940-02-26')
end = pd.to_datetime('1940-03-27')
original = original.sel(time=slice(start, end))
original_df = original['u10'].max(dim=['lat', 'lon']).to_dataframe('u10')

# %%
comparison.columns = ['before', 'after']
comparison['difference'] = comparison['after'] - comparison['before']
assert comparison['difference'].sum() == 0, 'Storm maxima do not match.'

# only 339 aligned at the moment
comparison = comparison.reset_index()
comparison['time'] = pd.to_datetime(comparison['time'])
# comparison.sort_values(by='time')
# comparison.groupby('time').agg(list)
# comparison.groupby('time').agg(max) # choses number over nans
comparison = comparison.groupby('time').agg(max) #? hope not losing any info
comparison[comparison['difference'].isnull()] # shows misaligned dates
num_misaligned_maxima = comparison['difference'].isnull().sum()
assert num_misaligned_maxima == 0, "Found {} misaligned storm maxima".format(num_misaligned_maxima)

# %%
comparison
# %%