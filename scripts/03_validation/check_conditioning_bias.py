"""
Identify conditioning grid cell for all storms.
"""
# %%
import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from environs import Env

SIMPLIFIED = False

env = Env()
env.read_env()

plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'figure.titlesize': 8,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'normal',
    'legend.fontsize': 6,
    'font.family': 'sans-serif'
})

datadir = env.str("PROCDIR")
figdir = env.str("FIGDIR")
path = os.path.join(datadir, "event_footprints.parquet")


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
    daily_path = os.path.join(datadir, "event_cubes.parquet")
    daily_df = pd.read_parquet(daily_path)
    daily_df.head()

    meta_path = os.path.join(datadir, "event_metadata.parquet")
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


# %% Add visualisations
path = os.path.join(datadir, "resampled_1941_2022.nc")
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
# %% === plot just the percentage of storms conditioned on each grid cell === 
fig, ax = plt.subplots(1, 1, figsize=(3, 3), subplot_kw={"projection": ccrs.PlateCarree()})
counts["perc"].plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap="OrRd",
                             cbar_kwargs={
                                 "label": "Conditioning frequency (%)",
                                 "orientation": "horizontal",
                                 "pad": 0.025, "shrink": 0.8,
                                 "format": lambda x, _: f"{x*100:.1f}"
                                })
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.8)
# ax.set_title("Number of storms conditioned on each grid cell (1941-2022)")

# %%
path = os.path.join(datadir, "resampled_1941_2022.nc")
ds = xr.open_dataset(path)
# %%
mean_u10 = ds["u10"].mean(dim="time")
median_u10 = ds["u10"].median(dim="time")
std_u10 = ds["u10"].std(dim="time")

anomaly_u10 = ds["u10"] - mean_u10
mean_anomaly_u10 = anomaly_u10.mean(dim="time")
std_anomaly_u10 = anomaly_u10.std(dim="time")
q99_anomaly_u10 = anomaly_u10.quantile(0.99, dim="time")



# %% plot mean, std and percentage of storms conditioned on each grid cell

# increase cbar font size
CMAP = "OrRd"

# resample counts to 1 degree grid for better visualisation
# comment to toggle resampling
# counts_resampled = counts["perc"].coarsen(lat=3, lon=3, boundary="trim").max()
counts_resampled = counts["perc"].copy()

fig, axs = plt.subplots(1, 3,
                       figsize=(4, 3),
                       subplot_kw={"projection": ccrs.PlateCarree()},
                       constrained_layout=True
                       )

mean_u10.plot(ax=axs[0], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={
                          "label": "mean u10 (m s⁻²)",
                          "orientation": "horizontal",
                          "pad": 0.01, "shrink": 0.8,
                          "format": lambda x, _: f"{x:.2f}",
                          "ticks": [0.0, 2.5, 5.0, 7.5]}
                        )

std_u10.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={
                          "label": "std u10 (m s⁻²)",
                          "orientation": "horizontal",
                          "pad": 0.01, "shrink": 0.8,
                          "format": lambda x, _: f"{x:.0f}",
                          "ticks": [1.0, 2.0, 3.0, 4.0]}
                )

counts_resampled.plot(ax=axs[2], transform=ccrs.PlateCarree(), cmap=CMAP,
                      cbar_kwargs={
                          "label": "freq. (%)",
                          "orientation": "horizontal",
                          "pad": 0.01, "shrink": 0.8,
                          "format": lambda x, _: f"{x*100:.1f}",
                          "ticks": [0.01, 0.02, 0.03]
                          })

for ax in axs.flatten():
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.8)

fig.savefig(os.path.join(figdir, "conditioning_bias.png"), dpi=300)

# %%

