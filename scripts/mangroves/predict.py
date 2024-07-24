"""
Load a pretrained model then plot and display EADs

---- To do ----
    - Make into functions
    - Do for training data
    - Do for hazGAN samples
    - Do for dependent/independent assumptions
"""
# %% ---- Setup ----
import os
import pandas as pd
import numpy as np
from joblib import load
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']
plt.style.use('bmh')

month = 7
run = 'clean-sweep-3__forpublish' #'clean-sweep-3'
yearly_rate = 18 # from R
wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')

warnings.warn("Need to add mangroves_with_slope.geojson and change agg func")

# %% ---- Load model and samples ----
modelpath = os.path.join(wd, 'results', 'mangroves', 'model__forpublish.pkl')
scalerpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'scaler.pkl')
with open(modelpath, 'rb') as f:
    model = load(f)

samplespath = os.path.join(wd, 'samples', f'{run}.nc')
samples = xr.open_dataset(samplespath)
samples.loc[{'channel': 'mslp'}]['anomaly'] = -samples.loc[{'channel': 'mslp'}].anomaly
month_medians = samples.sel(month=month).medians
samples_month = samples[['anomaly', 'dependent', 'independent']] + month_medians
samples_month = samples_month.stack(grid=['lat', 'lon'])

datapath = os.path.join(wd, 'training', 'res_18x22', 'data.nc')
data = xr.open_dataset(datapath)
data.loc[{'channel': 'mslp'}]['anomaly'] = -data.loc[{'channel': 'mslp'}].anomaly
data_month = data + month_medians # data['medians']
data_month = data_month.stack(grid=['lat', 'lon'])
train_month = data_month.isel(time=slice(0, 1000))
test_month = data_month.isel(time=slice(1000, None))

# %% ---- Predict (needs Pandas)----
def predict_damages(model, X: xr.DataArray, var: str) -> xr.DataArray:
    X = X[var].to_dataframe()[var]
    X = X.unstack('channel').reset_index()
    X.columns = ['sample', 'lat', 'lon', 'wind' ,'mslp']
    X = X.set_index(['sample', 'lat', 'lon'])
    X[['wind', 'mslp']] = X
    X[f'mangrove_damage'] = model.predict(X)
    damages = X[f'mangrove_damage'].to_xarray()
    return damages.to_dataset()

damages_dependent = predict_damages(model, samples_month, 'dependent')
damages_independent = predict_damages(model, samples_month, 'independent')
damages_hazGAN = predict_damages(model, samples_month, 'anomaly')
damages_train = predict_damages(model, train_month, 'anomaly')

# %% ---- Plot mangrove damage predictions for random storm ----
heatmap_kwargs = {'cmap': 'YlOrRd', 'cbar_kwargs': {'label': 'Mangrove damage (%area)'}}    

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
i = np.random.random_integers(0, damages_hazGAN.sizes['sample'])
damages_hazGAN.isel(sample=i).mangrove_damage.plot.contourf(ax=axs[0], levels=12, **heatmap_kwargs)
axs[0].set_title(f'Predicted mangrove damage (sample storm nᵒ {i})')

j = np.random.random_integers(0, damages_train.sizes['sample'])
damages_train.isel(sample=j).mangrove_damage.plot.contourf(ax=axs[1], levels=12, **heatmap_kwargs)
axs[1].set_title(f'Predicted mangrove damage (real storm nᵒ {j})')

# %% ---- Step 2: Load mangrove data ----
import cartopy.crs as ccrs
import geopandas as gpd 
from shapely.geometry import box
from hazGAN import xmin, xmax, ymin, ymax, bay_of_bengal_crs

aoi = box(xmin, ymin, xmax, ymax)
global_mangrove_watch = '/Users/alison/Documents/DPhil/data/gmw-v3-2020.nosync/gmw_v3_2020_vec.gpkg'
mangroves = gpd.read_file(global_mangrove_watch, mask=aoi)
mangroves = mangroves.set_crs(epsg=4326).drop(columns='PXLVAL')
mangroves['area'] = mangroves.to_crs(bay_of_bengal_crs).area
mangrove_centroids = mangroves.set_geometry(mangroves.centroid)
mangrove_centrois = mangrove_centroids.sort_values(by='area', ascending=False)

mangrovesout = os.path.join(wd, 'results', 'mangroves', 'mangroves.geojson')
mangroves.to_file(mangrovesout, driver='GeoJSON')

# %% ---- Convert mangroves to xarray.Dataset with values as areas ----
import xagg as xa

def intersect_mangroves_with_damages(mangroves: gpd.GeoDataFrame,
                                     damages: xr.Dataset,
                                     plot=True) -> xr.Dataset:
    # calculate intersections
    mangroves = mangroves.to_crs(4326)
    damages = damages_hazGAN.copy()
    weightmap = xa.pixel_overlaps(damages, mangroves)

    # calculate overlaps, NOTE: using EPSG:4326 for now
    mangroves_gridded = weightmap.agg
    mangroves_gridded['npix'] = mangroves_gridded['pix_idxs'].apply(len)
    mangroves_gridded['rel_area'] = mangroves_gridded['rel_area'].apply(lambda x: np.squeeze(x, axis=0))
    mangroves_gridded = mangroves_gridded.explode(['rel_area', 'pix_idxs'])

    # sum all relative mangrove areas in the same pixel
    mangroves_gridded['area'] = mangroves_gridded['area'] * mangroves_gridded['rel_area']
    mangroves_gridded = mangroves_gridded.groupby('pix_idxs').agg({'area': 'sum', 'coords': 'first'})

    # convert pd.DataFrame to xarray.Dataset
    lons = weightmap.source_grid['lon'].values
    lats = weightmap.source_grid['lat'].values
    mangroves_gridded = mangroves_gridded.reset_index()
    mangroves_gridded['lon'] = mangroves_gridded['pix_idxs'].apply(lambda j: lons[j])
    mangroves_gridded['lat'] = mangroves_gridded['pix_idxs'].apply(lambda i: lats[i])
    mangroves_gridded['lon'] = mangroves_gridded['lon'].astype(float)
    mangroves_gridded['lat'] = mangroves_gridded['lat'].astype(float)
    mangroves_gridded['area'] = mangroves_gridded['area'].astype(float)
    mangroves_gridded['area'] = mangroves_gridded['area'] * 1e-6 # convert to sqkm
    mangroves_gridded = mangroves_gridded.set_index(['lat', 'lon'])[['area']]
    mangroves_gridded = xr.Dataset.from_dataframe(mangroves_gridded)

    if plot:
        mangroves_gridded.area.plot(cmap="Greens", cbar_kwargs={'label': 'Mangrove damage [km²]'})
    return mangroves_gridded

mangroves_gridded = intersect_mangroves_with_damages(mangroves, damages_hazGAN)
mangroves_gridded.area.plot(cmap="Greens", cbar_kwargs={'label': 'Mangrove damage [km²]'})

damages_hazGAN['mangrove_damage_area'] = mangroves_gridded.area * damages_hazGAN.mangrove_damage
damages_hazGAN['mangrove_damage_percent'] = damages_hazGAN.mangrove_damage.where(mangroves_gridded.area > 0)

damages_dependent['mangrove_damage_area'] = mangroves_gridded.area * damages_dependent.mangrove_damage
damages_dependent['mangrove_damage_percent'] = damages_dependent.mangrove_damage.where(mangroves_gridded.area > 0)

damages_independent['mangrove_damage_area'] = mangroves_gridded.area * damages_independent.mangrove_damage
damages_independent['mangrove_damage_percent'] = damages_independent.mangrove_damage.where(mangroves_gridded.area > 0)

damages_train['mangrove_damage_area'] = mangroves_gridded.area * damages_train.mangrove_damage
damages_train['mangrove_damage_percent'] = damages_train.mangrove_damage.where(mangroves_gridded.area > 0)
# %% Turn contourf on and off
import cartopy.crs as ccrs
from cartopy import feature
from matplotlib.ticker import PercentFormatter
i = np.random.randint(0, 1000)

# i = 969
damages = damages_dependent
fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
mangroves_gridded.area.plot(ax=axs[0, 0], cmap="Greens", cbar_kwargs={'label': 'Mangrove area [km²]'})
damages.isel(sample=i).mangrove_damage.plot.contourf(ax=axs[0, 1],
                                                   cmap='YlOrRd',
                                                   cbar_kwargs={'label': 'Mangrove potential damage',
                                                                'format': PercentFormatter(1, 0)}
)
damages.isel(sample=i).mangrove_damage_percent.plot(ax=axs[1, 0],
                                                   cmap='YlOrRd',
                                                   cbar_kwargs={'label': 'Mangrove potential damage',
                                                                'format': PercentFormatter(1, 0)}
                                                                )

damages.isel(sample=i).mangrove_damage_area.plot(ax=axs[1, 1],
                                                        cmap='YlOrRd',
                                                        cbar_kwargs={'label': 'Mangrove damaged [km²]'}
                                                        )

axs[0, 0].set_title('Mangrove area')
axs[0, 1].set_title('Mangrove potential damage')
axs[1, 0].set_title('Mangrove percentage damage')
axs[1, 1].set_title('Mangrove damaged')

for ax in axs.ravel():
    ax.add_feature(feature.COASTLINE)
    ax.add_feature(feature.BORDERS)
    ax.add_feature(feature.LAND, facecolor='wheat')
    ax.add_feature(feature.OCEAN)

fig.suptitle('Sample storm damage to mangroves')

# %% ---- Calculate return periods and EADs ----
from sklearn.metrics import auc
from hazGAN import occurrence_rate


def auc_ufunc(x, y):
    x = sorted(x)
    y = sorted(y)
    return auc(x, y)


def calculate_eads(var, damages: xr.Dataset, yearly_rate: int) -> xr.Dataset:
    damages = damages.copy()
    exceedence_prob = 1 - (damages[var].rank(dim='sample') / (1 + damages[var].sizes['sample']))

    annual_exceedence_prob = yearly_rate * exceedence_prob
    return_period = 1 / annual_exceedence_prob
    damages['annual_exceedence_prob'] = annual_exceedence_prob
    damages['return_period'] = return_period

    damages['expected_annual_damages'] = xr.apply_ufunc(
        auc_ufunc,
        damages['annual_exceedence_prob'],
        damages[var],
        input_core_dims=[['sample'], ['sample']],
        vectorize=True
        )
    
    return damages

damages_dependent = calculate_eads('mangrove_damage_area', damages_dependent, occurrence_rate)
damages_independent = calculate_eads('mangrove_damage_area', damages_independent, occurrence_rate)
damages_hazGAN = calculate_eads('mangrove_damage_area', damages_hazGAN, occurrence_rate)
damages_train = calculate_eads('mangrove_damage_area', damages_train, occurrence_rate)
# %% ---- Plot EADs ----
fig, axs = plt.subplots(1, 4, figsize=(15, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})

damages_dependent.expected_annual_damages.plot(ax=axs[0], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_independent.expected_annual_damages.plot(ax=axs[1], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_hazGAN.expected_annual_damages.plot(ax=axs[2], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_train.expected_annual_damages.plot(ax=axs[3], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})

axs[0].set_title('All dependent')
axs[1].set_title('All independent')
axs[2].set_title('hazGAN')
axs[3].set_title('Real data')

# %% ---- Return period vs. damages plot (Lamb 2010) ----
def calculate_total_return_periods(damages: xr.Dataset,
                                   yearly_rate: float,
                                   var='mangrove_damage_area') -> xr.Dataset:
    totals = damages[var].sum(dim=['lat', 'lon']).to_dataset()
    N = totals[var].sizes['sample']
    totals['rank'] = totals[var].rank(dim='sample')
    totals['exceedence_probability'] = 1 - totals['rank'] / (1 + N)
    totals['return_period'] = 1 / (yearly_rate * totals['exceedence_probability'])
    totals = totals.sortby('return_period')
    return totals

totals_independent = calculate_total_return_periods(damages_independent, occurrence_rate)
totals_dependent = calculate_total_return_periods(damages_dependent, occurrence_rate)
totals_hazGAN = calculate_total_return_periods(damages_hazGAN, occurrence_rate)
totals_data = calculate_total_return_periods(damages_train, occurrence_rate)

# %%
fig, ax = plt.subplots()
eps =.25

mask = totals_hazGAN.return_period > eps
ax.plot(
    totals_hazGAN.where(mask).return_period,
    totals_hazGAN.where(mask).mangrove_damage_area,
    color='k',
    linewidth=1.5,
    label='Modelled dependence'
)

mask = totals_dependent.return_period > eps
ax.plot(
    totals_dependent.where(mask).return_period,
    totals_dependent.where(mask).mangrove_damage_area,
    color='blue',
    linewidth=1.5,
    linestyle='dotted',
    label='Complete dependence'
)

mask = totals_independent.return_period > eps
ax.plot(
    totals_independent.where(mask).return_period,
    totals_independent.where(mask).mangrove_damage_area,
    color='r',
    linestyle='dashed',
    linewidth=1.5,
    label='Independence'
)

mask = totals_data.return_period > eps
ax.scatter(
    totals_data.where(mask).return_period,
    totals_data.where(mask).mangrove_damage_area,
    color='k',
    s=1.5,
    label='Training data'
)

ax.set_xlabel('Return period (years)')
ax.set_ylabel('Total damage to mangroves (km²)')
ax.legend()
ax.set_xscale('log')
ax.set_xticks([2, 5, 25, 100, 200, 500])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title('Based on Figure 7 Lamb (2010)')
# %%
!say done
# %%
