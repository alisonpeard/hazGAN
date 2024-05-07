# %%
import os
os.environ["USE_PYGEOS"] = "0"
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from calendar import month_name as month

# %%
channels = ["u10", "mslp"]
wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = os.path.join(wd, "era5_data")
df = pd.read_parquet(os.path.join(datadir, f"fitted_data.parquet"))
coords = pd.read_parquet(os.path.join(datadir, "coords.parquet"))
coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"])).set_crs("EPSG:4326")
df = df.merge(coords, on="grid")
df.columns = [col.replace(".", "_") for col in df.columns]
df = df.rename(columns={"msl": "mslp"})
df["extremeness"] = df["extremeness_u10"]
df = df.drop(columns=["extremeness_u10", "extremeness_mslp"])

# add event sizes
events = (pd.read_parquet(os.path.join(datadir, "event_data.parquet"))[["cluster", "cluster.size"]].groupby("cluster").mean())
events = events.to_dict()["cluster.size"]
df["size"] = df["cluster"].map(events)
gdf = gpd.GeoDataFrame(df, geometry="geometry").set_crs("EPSG:4326")
# %%
c0 = gdf["cluster"].min()
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
var = "u10"

p_cmap = plt.get_cmap("viridis")
p_cmap.set_under("red")

gdf[gdf["cluster"] == c0].plot(column=f"p_{var}", marker="s", cmap=p_cmap, vmin=0.05, ax=axs[0])
gdf[gdf["cluster"] == c0].plot(column=f"thresh_{var}", legend=True, marker="s", cmap="viridis", ax=axs[1])
gdf[gdf["cluster"] == c0].plot(column=f"scale_{var}", legend=True, marker="s", cmap="viridis", ax=axs[2])
gdf[gdf["cluster"] == c0].plot(column=f"shape_{var}", legend=True, marker="s", cmap="viridis", ax=axs[3])

# extend p-values colorbar to show where H0 rejected
scatter = axs[0].collections[0]
plt.colorbar(scatter, ax=axs[0], extend="min")

axs[0].set_title("p, H0: GPD(ξ,μ,σ)")
axs[1].set_title("thresh (μ)")
axs[2].set_title("scale (σ)")
axs[3].set_title("shape (ξ)")
fig.suptitle(f"Fit for ERA5 {var.upper()}, N={gdf['cluster'].nunique()}")

print(gdf[gdf[f"p_{var}"] < 0.1]["grid"].nunique(), "significant p-values")
# %%
gdf["grid"].nunique()
fig = gdf[[f"p_{var}", "grid"]].groupby("grid").mean().hist(color="lightgrey", bins=15, edgecolor="k")
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


# %%
# use latitude and longitude columns to label grid points in (i,j) format
gdf["latitude"] = gdf["geometry"].apply(lambda x: x.y)
gdf["longitude"] = gdf["geometry"].apply(lambda x: x.x)
gdf = gdf.sort_values(["latitude", "longitude", "cluster"], ascending=[True, True, True])
gdf.head()
# %%
nchannels = len(channels)
# get dimensions
T = gdf["cluster"].nunique()
nx = gdf["longitude"].nunique()
ny = gdf["latitude"].nunique()

# make training tensors
X = gdf[channels].values.T.reshape([nchannels, ny, nx, T])[:, ::-1, :, :]
U = gdf[[f"ecdf_{c}" for c in channels]].values.T.reshape([nchannels, ny, nx, T])[:, ::-1, :, :]
M = gdf[[f"{c}_median" for c in channels]].values.T.reshape([nchannels, ny, nx, T])[:, ::-1, :, :]
X = np.swapaxes(X, 0, -1)
U = np.swapaxes(U, 0, -1)
M = np.swapaxes(M, 0, -1)
z = gdf[["cluster", "extremeness"]].groupby("cluster").mean().values.reshape([T])
t = gdf[["cluster"]].groupby("cluster").mean().reset_index().values.reshape([T])
s = gdf[["cluster", "size"]].groupby("cluster").mean().values.reshape([T])
# %%
i = np.random.uniform(0, T, 1).astype(int)[0]
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(U[i, ..., 0], vmin=0, vmax=1)
axs[0].set_title(f"Channel {0}")
axs[1].imshow(U[i, ..., 1], vmin=0, vmax=1)
axs[1].set_title(f"Channel {1}")
fig.suptitle(f"Pixel {i}")

vmin0 = M[i, ..., 0].min()
vmax0 = M[i, ..., 0].max()
vmin1 = M[i, ..., 1].min()
vmax1 = M[i, ..., 1].max()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(M[i, ..., 0], vmin=vmin0, vmax=vmax0)
axs[0].set_title(f"Median channel {0}")
axs[1].imshow(M[i, ..., 1], vmin=vmin1, vmax=vmax1)
axs[1].set_title(f"Median channel {1}")

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(M[i, ..., 0] + U[i, ..., 0], vmin=vmin0, vmax=vmax0)
axs[0].set_title(f"Median + anomaly channel {0}")
axs[1].imshow(M[i, ..., 1] + U[i, ..., 1])
axs[1].set_title(f"Median + anomaly channel {1}")

# %%
# parameters and coordinates (slightly hardcoded)
threshs = []
scales = []
shapes = []

if len(channels) > 1:
    gpd_params = ([f"thresh_{var}" for var in channels] + [f"scale_{var}" for var in channels] + [f"shape_{var}" for var in channels])
else:
    gpd_params = ["thresh", "scale", "shape"]

gdf_params = (gdf[[*gpd_params, "longitude", "latitude"]].groupby(["latitude", "longitude"]).mean().reset_index())

thresh = np.array(gdf_params[[f"thresh_{var}" for var in channels]].values.reshape([ny, nx, nchannels])[::-1, ...])
scale = np.array(gdf_params[[f"scale_{var}" for var in channels]].values.reshape([ny, nx, nchannels])[::-1, ...])
shape = np.array(gdf_params[[f"shape_{var}" for var in channels]].values.reshape([ny, nx, nchannels])[::-1, ...])

params = np.stack([shape, thresh, scale], axis=-2)
lat = gdf_params["latitude"].values.reshape([ny, nx])[::-1, ...]
lon = gdf_params["longitude"].values.reshape([ny, nx])[::-1, ...]
params.shape
# %%
np.savez(os.path.join(datadir, f"data.npz"), X=X, U=U, M=M, z=z,lat=lat, lon=lon, t=t, s=s, params=params)
# %% Save in netcdf format
import xarray as xr

events = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
times = pd.to_datetime(events[['cluster', 'time']].groupby('cluster').first()['time'].reset_index(drop=True))

ds = xr.Dataset({'U': (['time', 'lat', 'lon', 'channel'], U),
                 'X': (['time', 'lat', 'lon', 'channel'], X),
                 'M': (['time', 'lat', 'lon', 'channel'], M),
                 'z': (['time'], z),
                 's': (['time'], s),
                 'params': (['lat', 'lon', 'param', 'channel'], params)
                 },
                coords={'lat': (['lat'], lat[:, 0]),
                        'lon': (['lon'], lon[0, :]),
                        'time': times,
                        'channel': channels,
                        'param': ['shape', 'loc', 'scale']
                 },
                 attrs={'crs': 'EPSG:4326', 'u10': '10m wind speed', 'mslp': 'mean sea level pressure'})
ds.isel(time=1, channel=0).U.plot(cmap='Spectral_r')
ds.to_netcdf(os.path.join(datadir, "data.nc"))
# %%
