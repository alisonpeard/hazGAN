"""
Environment: general

Process CMIP6 data daily maximum wind speed at 10m.

References:
-----------
    https://carpentries-lab.github.io/python-aos-lesson/10-large-data/index.html
"""

# %%
import os
import warnings

os.environ["USE_PYGEOS"] = "0"
import glob
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def regrid(
    ds,
    xmin,
    xmax,
    ymin,
    ymax,
    nx,
    ny,
    method="conservative",
    extrap_method="nearest_s2d",
):
    ds_out = xr.Dataset(
        {
            "latitude": (["latitude"], np.linspace(ymin, ymax, ny)),
            "longitude": (["longitude"], np.linspace(xmin, xmax, nx)),
        }
    )
    regridder = xe.Regridder(ds, ds_out, method, extrap_method=extrap_method)
    ds = regridder(ds)
    return ds


def label_gridpoints(df):
    grid_ids = (
        df[["latitude", "longitude"]]
        .groupby(["latitude", "longitude"])
        .agg(sum)
        .reset_index()
    )
    grid_ids = {
        (row.latitude, row.longitude): row.Index for row in grid_ids.itertuples()
    }
    df["grid"] = df.apply(lambda row: grid_ids[(row.latitude, row.longitude)], axis=1)
    return df


wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = "/Users/alison/Documents/DPhil/data"
indir = os.path.join(datadir, "era5", "new_data")
outdir = os.path.join(wd, "era5_data")
# %%
# load all files and process to geopandas
files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))
ds = xr.open_mfdataset(files, chunks={"time": "500MB"}, engine="netcdf4")
ds["u10"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.drop(["v10"])

ds = regrid(ds, 80, 95, 10, 25, 22, 18, method="bilinear")
ds = ds.resample(time="1D").max()
df = ds.to_dataframe().reset_index()

warnings.warn("Removing msl values below 1000, this needs to be checked.")
df = df[df["msl"] > 1000]
df = df.dropna()

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"])
).set_crs("EPSG:4326")
# %%
# label grid points
try:
    # load coords
    print("Loading coords")
    coords = gpd.read_file(os.path.join(outdir, "coords.gpkg"))
    coords_dict = {
        (lat, lon): grid
        for grid, lat, lon in zip(
            coords["grid"], coords["latitude"], coords["longitude"]
        )
    }
    gdf["grid"] = gdf.apply(
        lambda row: coords_dict[(row.latitude, row.longitude)], axis=1
    )
except Exception as e:
    # make grid and save coords
    print(e)
    print("Making grid")
    gdf = label_gridpoints(gdf)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry").set_crs("EPSG:4326")
    coords = gdf[["grid", "latitude", "longitude", "geometry"]].drop_duplicates()
    coords = coords.set_index("grid").set_crs("EPSG:4326")
    coords.to_file(os.path.join(outdir, "coords.gpkg"), driver="GPKG")

# %%
# save daily maxima csv
year0 = gdf["time"].min().year
yearn = gdf["time"].max().year
gdf.drop(columns=["geometry", "longitude", "latitude"]).to_csv(
    os.path.join(outdir, f"data_{year0}_{yearn}.csv"), sep=",", index=False
)

# %%
# check looks okay over spatial domain
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# t0 = pd.to_datetime('2014-12-31') # gdf['time'].min()
t0 = pd.to_datetime("1951-12-04")
gdf[gdf["time"] == t0].plot(column="u10", legend=True, ax=axs[0])
gdf[gdf["time"] == t0].plot(column="msl", legend=True, ax=axs[1])
axs[0].set_title("10m wind")
axs[1].set_title("sea-level pressure")
fig.suptitle(t0)
