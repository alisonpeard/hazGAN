# %%
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import ecdf
import xesmf as xe

def regrid(ds, xmin, xmax, ymin, ymax, nx, ny, xvar='lon', yvar='lat', method='bilinear', extrap_method="nearest_s2d"):
    ds_out = xr.Dataset(
        {
            yvar: ([yvar], np.linspace(ymin, ymax, ny)),
            xvar: ([xvar], np.linspace(xmin, xmax, nx))
        }
    )
    regridder = xe.Regridder(ds, ds_out, method=method, extrap_method=extrap_method)
    ds = regridder(ds)
    return ds

datadir = os.path.join("/Users/alison/Documents/DPhil/data")
# %%
# Load ERA5
ds_era5 = xr.open_dataset(os.path.join(datadir, 'era5', 'wind_data', 'bangladesh_2013.nc'))
ds_era5['u10'] = np.sqrt(ds_era5['u10']**2 + ds_era5['v10']**2)
ds = ds_era5.drop(['v10'])
ds_era5 = ds_era5.resample(time='1D').mean()
ds_era5 = regrid(ds_era5, 80, 95, 10, 25, 22, 18, xvar='longitude', yvar='latitude', method='bilinear')
# %%
# Load CMIP6
files = glob.glob(os.path.join(datadir, 'cmip6', 'u10', f"*HadGEM3*.nc"))
ds_cmip6 = xr.open_mfdataset(files, chunks={'time': '500MB'}, decode_cf=False)
ds_cmip6 = xr.decode_cf(ds_cmip6)
ds_cmip6 = ds_cmip6.convert_calendar(calendar='gregorian', align_on='year', missing=np.nan)
ds_cmip6 = regrid(ds_cmip6, 80, 95, 10, 25, 22, 18, method='bilinear')
# %%
# match time periods
start_time = max(ds_era5.time.min(), ds_cmip6.time.min())
end_time = min(ds_era5.time.max(), ds_cmip6.time.max())
ds_era5 = ds_era5.sel(time=slice(start_time, end_time))
ds_cmip6 = ds_cmip6.sel(time=slice(start_time, end_time))
ds_cmip6 = ds_cmip6.interp(lat=ds_era5.latitude, lon=ds_era5.longitude, method='nearest')
# %%
gdf_era5 = ds_era5.to_dataframe().reset_index()
gdf_cmip6 = ds_cmip6.to_dataframe().reset_index()
# %%
gdf_era5['time'] = pd.to_datetime(gdf_era5['time'])
gdf_era5['day'] = gdf_era5['time'].dt.dayofyear
gdf_era5 = gdf_era5.groupby(['day', 'latitude', 'longitude']).mean() # mean not maximum wind
gdf_era5 = gdf_era5.reset_index()[['latitude', 'longitude', 'time', 'u10']]
gdf_era5 = gpd.GeoDataFrame(gdf_era5, geometry=gpd.points_from_xy(gdf_era5['longitude'], gdf_era5['latitude'])).set_crs('EPSG:4326')

# %%
gdf_cmip6 = gpd.GeoDataFrame(gdf_cmip6, geometry=gpd.points_from_xy(gdf_cmip6['lon'], gdf_cmip6['lat'])).set_crs('EPSG:4326')
gdf_cmip6 = gdf_cmip6[['time', 'sfcWind', 'geometry']]
gdf_cmip6 = gdf_cmip6.rename(columns={'sfcWind': 'u10'})
gdf_cmip6['time'] = gdf_cmip6['time'].dt.date
gdf_era5['time'] = gdf_era5['time'].dt.date
# %%
# preliminary visual comparison
t0 = gdf_era5['time'].min()

# heatmaps
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].set_title('ERA5')
axs[1].set_title('CMIP6')
fig.suptitle(f"Snapshot for {t0}")

vmin = gdf_era5[gdf_era5['time'] == t0]['u10'].min()
vmax = gdf_era5[gdf_era5['time'] == t0]['u10'].max()
gdf_era5[gdf_era5['time'] == t0].plot('u10', ax=axs[0], legend=True, vmin=vmin, vmax=vmax)
gdf_cmip6[gdf_cmip6['time'] == t0].plot('u10', ax=axs[1], legend=True, vmin=vmin, vmax=vmax);

# %%
# histograms
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].set_title('ERA5')
axs[1].set_title('CMIP6')
fig.suptitle(f"Snapshot for {t0}")
gdf_era5[gdf_era5['time'] == t0]['u10'].hist(ax=axs[0], color='lightgrey', edgecolor='k', bins=30, density=True)
gdf_cmip6[gdf_cmip6['time'] == t0]['u10'].hist(ax=axs[1], color='lightgrey', edgecolor='k', bins=30, density=True);
for ax in axs:
    ax.set_xlim(vmin, vmax)
# %%
# Tropical Cyclone Viyaru 10-15 May 2013
t0 = pd.to_datetime('2013-05-10')
tn = pd.to_datetime('2013-05-15')

gdf_era5['time'] = pd.to_datetime(gdf_era5['time'])
gdf_cmip6['time'] = pd.to_datetime(gdf_cmip6['time'])

fig, axs = plt.subplots(1, 1, figsize=(6, 3))
era5_ts = gdf_era5[(gdf_era5['time'] >= t0) & (gdf_era5['time'] <= tn)].groupby('time')['u10'].max()
cmip6_ts = gdf_cmip6[(gdf_cmip6['time'] >= t0) & (gdf_cmip6['time'] <= tn)].groupby('time')['u10'].max()

plt.plot(era5_ts, label='ERA5', color='blue')
plt.plot(cmip6_ts, label='CMIP6', color='orange')
plt.legend()
plt.title('Tropical Cyclone Viyaru 10-15 May 2013')

# %%
# statistics over the entire domain
maxima_era5 = gdf_era5.groupby('time')['u10'].max()[:-1]
maxima_cmip6 = gdf_cmip6.groupby('time')['u10'].max()
nan_inds = np.isnan(maxima_era5) | np.isnan(maxima_cmip6)
maxima_era5 = maxima_era5[~nan_inds]
maxima_cmip6 = maxima_cmip6[~nan_inds]

# Q-Q plots
quantiles_era5 = np.quantile(maxima_era5, np.linspace(0, 1, 100))
quantiles_cmip6 = np.quantile(maxima_cmip6, np.linspace(0, 1, 100))

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].scatter(quantiles_era5, quantiles_cmip6, color='k', s=1)
axs[0].plot(quantiles_era5, quantiles_era5, color='red', linestyle='dashed', linewidth=1)
axs[1].scatter(quantiles_cmip6, quantiles_era5, color='k', s=1)
axs[1].plot(quantiles_cmip6, quantiles_cmip6, color='red', linestyle='dashed', linewidth=1)
axs[0].set_title('ERA5 vs CMIP6')
axs[1].set_title('CMIP6 vs ERA5')
axs[0].set_xlabel('ERA5')
axs[0].set_ylabel('CMIP6')
axs[1].set_xlabel('CMIP6')
axs[1].set_ylabel('ERA5')

# %%
# Gumbel-Gumbel plot
def get_ecdf(series):
    return ecdf(series).cdf.evaluate(series)

maxima_era5 = maxima_era5[~nan_inds]
maxima_cmip6 = maxima_cmip6[~nan_inds]

ecdf_era5 = get_ecdf(maxima_era5.dropna())
ecdf_cmip6 = get_ecdf(maxima_cmip6.dropna())

gumbel_era5 = -np.log(-np.log(ecdf_era5))
gumbel_cmip6 = -np.log(-np.log(ecdf_cmip6))

plt.scatter(sorted(gumbel_era5), sorted(gumbel_cmip6), color='k', s=1)
plt.plot(gumbel_era5, gumbel_era5, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("ERA5")
plt.ylabel("CMIP6")
plt.title('Gumbel-Gumbel plot');
# %%
# merge GeoDataFrames to do proper spatial comparison
merged = pd.merge(gdf_era5, gdf_cmip6, how='inner', on=['time', 'geometry'], suffixes=('_era5', '_cmip6'))
merged['diff'] = merged['u10_era5'] - merged['u10_cmip6']
merged = merged.dropna()
# %%
# spatial analysis
aggfun = np.mean
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
spatial = merged.groupby('geometry')[['u10_era5', 'u10_cmip6', 'diff']].agg(aggfun).reset_index()
spatial = gpd.GeoDataFrame(spatial, geometry='geometry').set_crs('EPSG:4326')

vmin = spatial['u10_era5'].min()
vmax = spatial['u10_era5'].max()

spatial.plot('u10_era5', legend=True, cmap='Spectral_r', ax=axs[0], vmin=vmin, vmax=vmax)
spatial.plot('u10_cmip6', legend=True, cmap='Spectral_r', ax=axs[1], vmin=vmin, vmax=vmax)
spatial.plot('diff', legend=True, cmap='Spectral_r', ax=axs[2])
axs[0].set_title('ERA5')
axs[1].set_title('CMIP6')
axs[2].set_title('ERA5 - CMIP6')
fig.suptitle('Average over time');

# %%
# plot correlations
fig, axs = plt.subplots(1, 1, figsize=(3, 3))
correlations = merged[['geometry', 'u10_era5', 'u10_cmip6']].groupby('geometry').apply(lambda row: row['u10_era5'].corr(row['u10_cmip6'])).reset_index()
correlations = gpd.GeoDataFrame(correlations, geometry='geometry').set_crs('EPSG:4326')
correlations = correlations.rename(columns={0: 'correlation'})
correlations.plot('correlation', legend=True, cmap='Spectral_r', ax=axs)
fig.suptitle('Correlation between ERA5 and CMIP6');
# %%
# extremal dependence
...
gdf_era5

# %%
pivoted = gdf_era5[['time', 'geometry', 'u10']].pivot(index='time', columns='geometry').apply(get_ecdf, axis=0)
pivoted = pivoted.reorder_levels(['geometry', None], axis=1)
ecdf_era5 = pivoted.reset_index().melt(id_vars='time', var_name=['geometry'], value_name='u10')
ecdf_era5 = gpd.GeoDataFrame(ecdf_era5, geometry='geometry').set_crs(4326)

pivoted = gdf_cmip6[['time', 'geometry', 'u10']].dropna(subset='u10').pivot(index='time', columns='geometry').apply(get_ecdf, axis=0)
pivoted = pivoted.reorder_levels(['geometry', None], axis=1)
ecdf_cmip6 = pivoted.reset_index().melt(id_vars='time', var_name=['geometry'], value_name='u10')
ecdf_cmip6 = gpd.GeoDataFrame(ecdf_cmip6, geometry='geometry').set_crs(4326)

fig, axs = plt.subplots(1, 2, figsize=(4, 2))
ecdf_era5[ecdf_era5['time'] == t0].plot('u10', ax=axs[0], legend=True)
ecdf_cmip6[ecdf_cmip6['time'] == t0].plot('u10', ax=axs[1], legend=True)

# %%
ecdf_era5['inv_frechet'] = -np.log(ecdf_era5['u10'])    
ecdf_cmip6['inv_frechet'] = -np.log(ecdf_cmip6['u10'])

# %%
frechet = pd.merge(ecdf_era5, ecdf_cmip6, on=['geometry', 'time'], how='inner', suffixes=('_era5', '_cmip6'))
frechet['min'] = frechet[['inv_frechet_era5', 'inv_frechet_cmip6']].apply(min, axis=1)
frechet = frechet[['geometry', 'min']].groupby('geometry').agg(lambda x: len(x) / sum(x))
frechet = gpd.GeoDataFrame(frechet.reset_index(), geometry='geometry').set_crs(4326)

fig, axs = plt.subplots(1, 1, figsize=(3, 3))
frechet.plot('min', legend=True, ax=axs, cmap='Spectral')
fig.suptitle('Extremal coefficient ERA5 and CMIP6')
# %%
