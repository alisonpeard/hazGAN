# %%
import os
os.environ["USE_PYGEOS"] = "0"
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from calendar import month_name as month

# %%
channels = ["u10", "mslp"]
wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = os.path.join(wd, "era5_data")

coords = xr.open_dataset('/Users/alison/Documents/DPhil/multivariate/era5_data/data_1950_2022.nc')
coords = coords['grid'].to_dataframe().reset_index()
coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords['longitude'], coords['latitude'])).set_crs("EPSG:4326")
                                                                                                                
df = pd.read_parquet(os.path.join(datadir, f"fitted_data.parquet"))
df = df.merge(coords, on="grid")
df.columns = [col.replace(".", "_") for col in df.columns]
df = df.rename(columns={"msl": "mslp"})
df["storm_rp"] = df["storm_rp_u10"]
df = df.drop(columns=["storm_rp_u10", "storm_rp_mslp"])

# add event sizes
events = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))[["storm", "storm.size"]].groupby("storm").mean()
events = events.to_dict()["storm.size"]
df["size"] = df["storm"].map(events)
gdf = gpd.GeoDataFrame(df, geometry="geometry").set_crs("EPSG:4326")
# %%
p_crit = 0.1
s0 = gdf["storm"].min()
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
var = "u10"

p_cmap = plt.get_cmap("viridis")
p_cmap.set_under("red")

gdf[gdf["storm"] == s0].plot(column=f"p_{var}", marker="s", cmap=p_cmap, vmin=p_crit, ax=axs[0])

gdf[gdf["storm"] == s0].plot(column=f"thresh_{var}", legend=True, marker="s", cmap="viridis", ax=axs[1])
gdf[gdf["storm"] == s0].plot(column=f"scale_{var}", legend=True, marker="s", cmap="viridis", ax=axs[2])
gdf[gdf["storm"] == s0].plot(column=f"shape_{var}", legend=True, marker="s", cmap="viridis", ax=axs[3])

# extend p-values colorbar to show where H0 rejected
scatter = axs[0].collections[0]
plt.colorbar(scatter, ax=axs[0], extend="min")

axs[0].set_title("p, H0: GPD(ξ,μ,σ)")
axs[1].set_title("thresh (μ)")
axs[2].set_title("scale (σ)")
axs[3].set_title("shape (ξ)")
fig.suptitle(f"Fit for ERA5 {var.upper()}, N={gdf['storm'].nunique()}")

print(gdf[gdf[f"p_{var}"] < p_crit]["grid"].nunique(), "significant p-values")
# %%
monthly_medians = pd.read_csv(os.path.join(datadir, "monthly_medians.csv"), index_col="month")
assert monthly_medians.groupby(['month', 'grid']).count().max().max() == 1, "Monthly medians not unique"
monthly_medians = monthly_medians.groupby(["month", "grid"]).mean().reset_index()

for var in channels:
    gdf[f"month_{var}"] = pd.to_datetime(gdf[f"time_{var}"]).dt.month.map(lambda x: month[x])
    n = len(gdf)
    gdf = gdf.join(monthly_medians[['month', 'grid', var]].set_index(["month", "grid"]), on=[f"month_{var}", "grid"], rsuffix="_median")
    assert n == len(gdf), "Merge failed"
    del gdf[f'month_{var}']

# use latitude and longitude columns to label grid points in (i,j) format
gdf["latitude"] = gdf["geometry"].apply(lambda x: x.y)
gdf["longitude"] = gdf["geometry"].apply(lambda x: x.x)
gdf = gdf.sort_values(["latitude", "longitude", "storm"], ascending=[True, True, True])
gdf.head()
# %% make netcdf file
nchannels = len(channels)
T = gdf["storm"].nunique()
nx = gdf["longitude"].nunique()
ny = gdf["latitude"].nunique()

# make training tensors
events = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
times = pd.to_datetime(events[['storm', 'time']].groupby('storm').first()['time'].reset_index(drop=True))
gdf = gdf.sort_values(["storm", "latitude", "longitude"], ascending=[True, True, True]) # [T, i, j, channel]
grid = gdf["grid"].unique().reshape([ny, nx])
lat = gdf["latitude"].unique()
lon = gdf["longitude"].unique()
X = gdf[channels].values.reshape([T, ny, nx, nchannels])
U = gdf[[f"ecdf_{c}" for c in channels]].values.reshape([T, ny, nx, nchannels])
M = gdf[[f"{c}_median" for c in channels]].values.reshape([T, ny, nx, nchannels])
z = gdf[["storm", "storm_rp"]].groupby("storm").mean().values.reshape(T)
s = gdf[["storm", "size"]].groupby("storm").mean().values.reshape(T)

lifetime_max_wind = np.max((X + M)[..., 0], axis=(1,2))
lifetime_min_pressure = np.min(-(X + M)[..., 1], axis=(1,2))

# %% parameters
threshs = []
scales = []
shapes = []
if len(channels) > 1:
    gpd_params = ([f"thresh_{var}" for var in channels] + [f"scale_{var}" for var in channels] + [f"shape_{var}" for var in channels])
else:
    gpd_params = ["thresh", "scale", "shape"]
gdf_params = (gdf[[*gpd_params, "longitude", "latitude"]].groupby(["latitude", "longitude"]).mean().reset_index())
thresh = np.array(gdf_params[[f"thresh_{var}" for var in channels]].values.reshape([ny, nx, nchannels]))
scale = np.array(gdf_params[[f"scale_{var}" for var in channels]].values.reshape([ny, nx, nchannels]))
shape = np.array(gdf_params[[f"shape_{var}" for var in channels]].values.reshape([ny, nx, nchannels]))
params = np.stack([shape, thresh, scale], axis=-2)

# %%
ds = xr.Dataset({'uniform': (['time', 'lat', 'lon', 'channel'], U),
                 'anomaly': (['time', 'lat', 'lon', 'channel'], X),
                 'medians': (['time', 'lat', 'lon', 'channel'], M),
                 'storm_rp': (['time'], z),
                 'duration': (['time'], s),
                 'params': (['lat', 'lon', 'param', 'channel'], params),
                 },
                coords={'lat': (['lat'], lat),
                        'lon': (['lon'], lon),
                        'time': times,
                        'channel': channels,
                        'param': ['shape', 'loc', 'scale']
                 },
                 attrs={'CRS': 'EPSG:4326', 'u10': '10m wind speed', 'mslp': 'negative of mean sea level pressure'})

ds.to_netcdf(os.path.join(datadir, "data.nc"))
# %% view netcdf file
t = np.random.uniform(0, T, 1).astype(int)[0]
ds_t = ds.isel(time=t)
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
ds_t.isel(channel=0).anomaly.plot(cmap='viridis', ax=axs[0])
ds_t.isel(channel=1).anomaly.plot(cmap='viridis', ax=axs[1])
fig.suptitle("Anomaly")

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
ds_t.isel(channel=0).medians.plot(cmap='viridis', ax=axs[0])
ds_t.isel(channel=1).medians.plot(cmap='viridis', ax=axs[1])
fig.suptitle(f"Median")

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
(ds_t.isel(channel=0).anomaly + ds_t.isel(channel=0).medians).plot(cmap='viridis', ax=axs[0])
(ds_t.isel(channel=1).anomaly + ds_t.isel(channel=1).medians).plot(cmap='viridis', ax=axs[1])
fig.suptitle('Anomaly + Median')
# %%
ds.close()
# %% check the highest wind speed is also the highest return period
highest_wind = ds.anomaly.isel(channel=0).max(dim=['lat', 'lon']).values.max()
highest_rp_wind = ds.anomaly.isel(channel=0, time=ds.storm_rp.argmax()).max(dim=['lat', 'lon']).values
assert highest_wind == highest_rp_wind, "Highest wind speed doesn't correspond to highest return period"

# %% have a look at the highest return period event
ds_outlier = ds.isel(time=ds.storm_rp.argmax())
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
wind_footprint = ds_outlier.anomaly.isel(channel=0) + ds_outlier.medians.isel(channel=0)
pressure_footprint = -(ds_outlier.anomaly.isel(channel=1) + ds_outlier.medians.isel(channel=1))
wind_footprint.plot(cmap='viridis', ax=axs[0])
pressure_footprint.plot(cmap='viridis', ax=axs[1])
# %%
plt.hist(ds.isel(channel=0).anomaly.values.ravel());
ds.close()
#%%