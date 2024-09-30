"""
Compare IBTrACs, ERA5, and IMDAA wind speeds and sea-level pressure estimates for Cyclone Amphan.
"""
# %%
import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
#%% Compare to IBTrACS hurricane records
xmin = 80
xmax = 95
ymin = 10
ymax = 25
IBTRACS_AGENCY_10MIN_WIND_FACTOR = {"wmo": [1.0, 0.0],
                                    "usa": [1.0, 0.0], "tokyo": [1.0, 0.0],
                                    "newdelhi": [0.88, 0.0], "reunion": [1.0, 0.0],
                                    "bom": [1.0, 0.0], "nadi": [1.0, 0.0],
                                    "wellington": [1.0, 0.0], 'cma': [1.0, 0.0],
                                    'hko': [1.0, 0.0], 'ds824': [1.0, 0.0],
                                    'td6': [1.0, 0.0], 'td5': [1.0, 0.0],
                                    'neumann': [1.0, 0.0], 'mlc': [1.0, 0.0],
                                    'hurdat_atl' : [0.88, 0.0], 'hurdat_epa' : [0.88, 0.0],
                                    'atcf' : [0.88, 0.0],     'cphc': [0.88, 0.0]
}

ibtracs = gpd.read_file(os.path.expandvars("$HOME/Documents/DPhil/data/ibtracs/ibtracs_since1980_points.gpkg"),
                        bbox=[xmin, ymin, xmax, ymax])
ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
wind_cols = [col for col in ibtracs.columns if 'WIND' in col.upper()]
pressure_cols = [col for col in ibtracs.columns if 'PRES' in col.upper()]
rmw_col = 'USA_RMW'
factors = {key[:3]: value for key, value in IBTRACS_AGENCY_10MIN_WIND_FACTOR.items()}
for col in wind_cols:
    agency = col.split('_')[0].lower()
    scale, shift = factors[agency]
    ibtracs[col] = ibtracs[col] * scale + shift
ibtracs['wind'] = ibtracs[wind_cols].max(axis=1) * 0.514 # knots to mps
ibtracs['pressure'] = ibtracs[pressure_cols].min(axis=1)
ibtracs['rmw'] = ibtracs[rmw_col]
ibtracs = ibtracs[['time', 'NAME', 'wind', 'pressure', 'rmw', 'LAT', 'LON']]
ibtracs.columns = ['time', 'event', 'wind', 'pressure', 'rmw', 'lat', 'lon']
ibtracs = ibtracs.groupby(pd.Grouper(key='time', freq='D')).agg({'event': 'first', 'wind': 'max',
                                                                 'rmw': 'max',
                                                                 'pressure': 'min', 'lat': 'mean',
                                                                 'lon': 'mean'}).reset_index()
ibtracs['event'] = ibtracs['event'].fillna('NOT_NAMED').apply(lambda s: s.title())
ibtracs = ibtracs.dropna(subset=['wind'])

CYCLONE = 'Asani'
cyclone = ibtracs[ibtracs['event'] == CYCLONE]

cyclone = gpd.GeoDataFrame(cyclone, geometry=gpd.points_from_xy(cyclone.lon, cyclone.lat)).drop(columns=['lat', 'lon'])
cyclone = cyclone.set_crs(epsg=4326)

# for selecting ERA5 and IMDAA
t0 = cyclone['time'].min()
t1 = cyclone['time'].max()
year = cyclone['time'].dt.year.unique()[0]

cyclone.head()

# %% # ERA5
path = f'/Users/alison/Documents/DPhil/data/era5/bangladesh_{year}.nc'
era5 = xr.open_dataset(path)
era5 = era5.sel(time=slice(t0, t1))
era5['u10'] = np.sqrt(era5['u10']**2 + era5['v10']**2)
era5 = era5.drop_vars(['v10'])
era5_footprint = era5.max(dim='time')

vmin = era5_footprint['u10'].quantile(0.01).values
vmax = era5_footprint['u10'].quantile(0.99).values

# %% plot era5 daily maxima over storm
for i, t in enumerate(np.arange(t0, t1 + pd.Timedelta(days=1), pd.Timedelta(days=1))):
    fig, ax = plt.subplots(figsize=(3, 3.1))
    era5.isel(time=i).u10.plot(ax=ax, cmap="RdBu_r", add_colorbar=False, vmin=vmin, vmax=vmax)
    ax.set_title(f'Day {i+1}')
    ax.axis('off')
    #plt.savefig(os.path.join("/Users/alison/Documents/DPhil/paper1.nosync/figures/paper", CYCLONE, f'day{i}.png'), transparent=True)

# %% plot footprint

fig, ax = plt.subplots(1, 1, figsize=(3, 3.1))
era5_footprint['u10'].plot(ax=ax, cmap='RdBu_r', add_colorbar=False, vmin=vmin, vmax=vmax)
ax.axis('off')
ax.set_title(f"TC {CYCLONE}")
plt.savefig(os.path.join("/Users/alison/Documents/DPhil/paper1.nosync/figures/paper", CYCLONE.lower(), f'footprint.png'), transparent=True)

# %%
imdaa_u10 = "/Users/alison/Documents/DPhil/data/imdaa/new_data/ncum_imdaa_reanl_HR_UGRD-10m_2020050100-2020053123.nc"
imdaa_v10 = "/Users/alison/Documents/DPhil/data/imdaa/new_data/ncum_imdaa_reanl_HR_VGRD-10m_2020050100-2020053123.nc"
imdaa_msl = "/Users/alison/Documents/DPhil/data/imdaa/new_data/ncum_imdaa_reanl_HR_PRMSL-msl_2020050100-2020053123.nc"

imdaa_u10 = xr.open_dataset(imdaa_u10)
imdaa_v10 = xr.open_dataset(imdaa_v10)
imdaa_msl = xr.open_dataset(imdaa_msl)

imdaa = xr.merge([imdaa_u10, imdaa_v10, imdaa_msl])
imdaa = imdaa.rename_vars({'UGRD_10m': 'u10', 'VGRD_10m': 'v10', 'PRMSL_msl': 'msl'})
imdaa = imdaa.sel(longitude=slice(xmin, xmax), latitude=slice(ymin, ymax))
imdaa = imdaa.sel(time=slice(t0, t1))
imdaa['u10'] = np.sqrt(imdaa['u10']**2 + imdaa['v10']**2)
imdaa = imdaa.drop_vars(['v10'])
imdaa_footprint = imdaa.max(dim='time')
# %% # plot comparison
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

era5_footprint['u10'].plot(ax=axs[0])
imdaa_footprint['u10'].plot(ax=axs[1])

# %%
imdaa_footprint['u10'].max().values, era5_footprint['u10'].max().values
# %%
# ibtracs
import geospatial_utils as gu

coords = era5_footprint.to_dataframe().reset_index()
coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.longitude, coords.latitude)).set_crs(4326)
units_df = pd.DataFrame.from_dict({'wind':['mps'], 'rmw': ['nmile'], 'pressure': ['hPa']}, orient='columns')
cyclone['ISO_TIME'] = cyclone['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
cyclone['BASIN'] = 'NI'
cyclone = cyclone.reset_index(drop=True)
cyclone['LAT'] = cyclone['geometry'].y
cyclone['LON'] = cyclone['geometry'].x
field = gu.get_wind_field(cyclone, coords, units_df, 'wind', 'pressure', 'rmw')
# %%
wind_cols = [col for col in field.columns if 'wnd' in col]
pressure_cols = [col for col in field.columns if 'pres' in col]
field['u10'] = field[wind_cols].max(axis=1)
field = field.drop(columns=wind_cols)
field = field.drop(columns=pressure_cols)
# %%
field.plot('u10', legend=True)

# %%
field = field.drop(columns='geometry').set_index(['latitude', 'longitude'])
ibtracs_footprint = xr.Dataset.from_dataframe(field)
ibtracs_footprint
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(14, 4))

era5_footprint['u10'].plot(ax=axs[0])
imdaa_footprint['u10'].plot(ax=axs[1])
ibtracs_footprint['u10'].plot(ax=axs[2])

axs[0].set_title('ERA5 (hourly)')
axs[1].set_title('IMDAA (hourly)')
axs[2].set_title('IBTrACS (6-hourly)')
fig.suptitle(f'Comparison of ERA5, IMDAA, and IBTrACS wind speeds for Cyclone {CYCLONE}')
# %%
