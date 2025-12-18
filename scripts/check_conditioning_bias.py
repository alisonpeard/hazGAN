"""
This is a bit over-simplified, it identifies the maximum wind speed per storm rather
than the daily max grid cell over the storm duration, but should be sufficient to
get an idea of the conditioning bias.
"""
# %%
import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SIMPLIFIED = False

path = "/Users/alison/Documents/dphil/paper1/nhess/v1/hazGAN-data/processing/storms.parquet"


if SIMPLIFIED:
    # just identify the grid cell with max wind speed per storm
    df = pd.read_parquet(path)
    print(f"{df.shape[0] / 4096} storms")
    df.head()

    def conditioning_gridcell(df):
        u10 = df["u10"]
        idxmax = u10.idxmax()
        return df.loc[idxmax, "grid"]

    grids = df.groupby("storm").apply(conditioning_gridcell)
    nevents = len(grids)
    grid_counts = grids.value_counts()
    grid_counts = pd.DataFrame(grid_counts)
    grid_counts = grid_counts.reset_index(drop=False, names=["grid", "counts"])
    grid_counts["grid"] = grid_counts["grid"].astype(int)

else:
    # try including all days of storms
    os.listdir(os.path.dirname(path))
    daily_path = "/Users/alison/Documents/dphil/paper1/nhess/v1/hazGAN-data/processing/daily.parquet"
    daily_df = pd.read_parquet(daily_path)
    daily_df.head()

    meta_path = "/Users/alison/Documents/dphil/paper1/nhess/v1/hazGAN-data/processing/storms_metadata.parquet"
    meta_df = pd.read_parquet(meta_path)
    meta_df.head()

    nevents = meta_df["storm"].nunique()

    # left-join and meta and daily on time column
    # keep storm and u10 from meta and u10 from daily
    # only keep rows where u10 from meta and daily are equal
    merged_df = pd.merge(meta_df[["storm", "time", "u10"]], daily_df[["grid","time", "u10"]],
                            on="time", suffixes=("_meta", "_daily"))
    filtered_df = merged_df[merged_df["u10_meta"] == merged_df["u10_daily"]]
    print(f"Number of matching records: {filtered_df.shape[0]} out of {merged_df.shape[0]}")
    grid_counts = filtered_df.groupby("grid").size().reset_index(name="count").sort_values(by="count", ascending=False)


# %% Add visualisation
path = "/Users/alison/Documents/dphil/paper1/nhess/v1/hazGAN-data/processing/data_1941_2022.nc"
coords = xr.open_dataset(path)
coords = coords['grid'].to_dataframe().reset_index()
coords = gpd.GeoDataFrame(
    coords, geometry=gpd.points_from_xy(coords['lon'], coords['lat'])
).set_crs("EPSG:4326")
coords["grid"] = coords["grid"].astype(int)                                                                            

# %%
df = grid_counts.merge(coords, on="grid", how="right")
gdf = gpd.GeoDataFrame(df, geometry="geometry")
gdf = gdf.fillna(0.0)
gdf = gdf.set_crs("EPSG:4326")

print(f"Max number of storms conditioned on a single grid cell: {gdf['count'].max()} out of {nevents} total storms")

counts = gdf.set_index(["lat", "lon"]).to_xarray()
counts["perc"] = counts["count"] / nevents

print(f"Max conditioning percentage: {counts['perc'].max().item()*100:.2f}%")
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
counts["perc"].plot(ax=ax, transform=ccrs.PlateCarree(), cmap="OrRd")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title("Number of storms conditioned on each grid cell (1941-2022)")

# %%
path = "/Users/alison/Documents/dphil/paper1/nhess/v1/hazGAN-data/processing/data_1941_2022.nc"
ds = xr.open_dataset(path)
# %%
mean_u10 = ds["u10"].mean(dim="time")
median_u10 = ds["u10"].median(dim="time")
std_u10 = ds["u10"].std(dim="time")

anomaly_u10 = ds["u10"] - mean_u10
mean_anomaly_u10 = anomaly_u10.mean(dim="time")
std_anomaly_u10 = anomaly_u10.std(dim="time")
q99_anomaly_u10 = anomaly_u10.quantile(0.99, dim="time")

plt.rcParams.update({'font.size': 8})

# increase cbar font size
CMAP = "OrRd"

fig, ax = plt.subplots(1, 3, figsize=(11/2, 5/2), subplot_kw={"projection": ccrs.PlateCarree()})
mean_u10.plot(ax=ax[0], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={"label": "m/s", "orientation": "horizontal", "pad": 0.01, "shrink": 0.9,
                                   "format": lambda x, _: f"{x:.2f}"})
ax[0].add_feature(cfeature.COASTLINE)
ax[0].add_feature(cfeature.BORDERS, linestyle=':')
ax[0].set_title("Mean 10 m wind speed")

std_u10.plot(ax=ax[1], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={"label": "m/s", "orientation": "horizontal", "pad": 0.01, "shrink": 0.9,
                                   "format": lambda x, _: f"{x:.3f}"})

ax[1].add_feature(cfeature.COASTLINE)
ax[1].add_feature(cfeature.BORDERS, linestyle=':')
ax[1].set_title("Std. of 10 m wind speed ")

counts["perc"].plot(ax=ax[2], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={"label": "conditioning frequency (%)",
                                   "orientation": "horizontal", "pad": 0.01,
                                   "shrink": 0.9,
                                   "format": lambda x, _: f"{x*100:.1f}"
                                   })
ax[2].add_feature(cfeature.COASTLINE)
ax[2].add_feature(cfeature.BORDERS, linestyle=':')
ax[2].set_title("Storm conditioning")

# fig.suptitle("1941-2022")
plt.tight_layout()
# %%

