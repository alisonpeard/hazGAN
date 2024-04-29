"""Process CMIP6 data daily maximum wind speed at 10m.

References:
-----------
    https://carpentries-lab.github.io/python-aos-lesson/10-large-data/index.html
"""

# %%
import os
os.environ['USE_PYGEOS'] = '0'
import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import geopandas as gpd
import matplotlib.pyplot as plt


def regrid(ds, xmin, xmax, ymin, ymax, nx, ny, method='bilinear', extrap_method="nearest_s2d"):
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(ymin, ymax, ny)),
            "lon": (["lon"], np.linspace(xmin, xmax, nx)),
        }
    )
    regridder = xe.Regridder(ds, ds_out, method, extrap_method=extrap_method)
    ds = regridder(ds)
    return ds

def label_gridpoints(df):
    grid_ids = df[['latitude', 'longitude']].groupby(['latitude', 'longitude']).agg(sum).reset_index()
    grid_ids = {(row.latitude, row.longitude): row.Index for row in grid_ids.itertuples()}
    df['grid'] = df.apply(lambda row: grid_ids[(row.latitude, row.longitude)], axis=1)
    return df

invar = "sfcWind" # name of variable in input files
var = "u10"  # name of variable to be output

model = "HadGEM3-GC31-LL"
wd = "/Users/alison/Documents/DPhil/multivariate"
datadir = "/Users/alison/Documents/DPhil/data"
indir = os.path.join(datadir, "cmip6", var)
outdir = os.path.join(wd, "cmip6_data")

# %%
# glob for all files
files = glob.glob(os.path.join(indir, f"*{model}*.nc"))
ds = xr.open_mfdataset(files, chunks={'time': '500MB'}, decode_cf=False)
ds = xr.decode_cf(ds)
ds = ds.convert_calendar(calendar='gregorian', align_on='year', missing=np.nan)
ds = regrid(ds, 80, 95, 10, 25, 22, 18, method='bilinear')
ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
# %%
# process dataset
df = ds.to_dataframe().reset_index()
df = df.dropna()
df = df[['time', invar, 'latitude', 'longitude']]
df.columns = ['time', var, 'latitude', 'longitude']
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).set_crs('EPSG:4326')

# %%
# label grid points
try:
    # load coords
    print("Loading coords")
    coords = gpd.read_file(os.path.join(outdir, "coords.gpkg"))
    coords_dict = {(lat, lon): grid for grid, lat, lon in zip(coords['grid'], coords['latitude'], coords['longitude'])}
    gdf['grid'] = gdf.apply(lambda row: coords_dict[(row.latitude, row.longitude)], axis=1)
except Exception as e:
    # make grid and save coords
    print("Making grid")
    gdf = label_gridpoints(gdf)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry').set_crs('EPSG:4326')
    coords = gdf[['grid', 'latitude', 'longitude', 'geometry']].drop_duplicates()
    coords = coords.set_index('grid').set_crs('EPSG:4326')
    coords.to_file(os.path.join(outdir, "coords.gpkg"), driver="GPKG")

# %%
# save daily maxima csv
year0 = gdf['time'].min().year
yearn = gdf['time'].max().year
gdf[['grid', 'time', var]].to_csv(os.path.join(outdir, f"{var}_{year0}_{yearn}.csv"), sep=',', index=False)

# %%
# check looks okay over spatial domain
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
t0 = pd.to_datetime('2014-12-31 12:00:00')
gdf[gdf['time'] == t0].plot(column='u10', legend=True, ax=ax, vmax=16)
ax.set_title(t0)

# %%
