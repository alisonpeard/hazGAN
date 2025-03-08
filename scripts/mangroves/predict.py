"""
Load a pretrained model then plot and display EADs

---- To do ----
    - Tidy up and refactor
"""
# %% user vars
month     = 7
DEV       = False
SMOOTHING = False
PLOT      = False
run       = 'amber-sweep-13' # 'vital-sweep-30__precipsota'
wd        = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')

# %% ---- Setup ----
import os
import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
# from joblib import load
import xarray as xr
# import xagg as xa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cartopy.crs as ccrs
from cartopy import feature
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
import warnings

from hazGAN import (
    xmin, xmax, ymin, ymax,
    bay_of_bengal_crs,
    occurrence_rate
)
from hazGAN import mangrove_example as mangroves

# global model
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']
plt.style.use('bmh')

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    config = {key: value["value"] for key, value in config.items()}
    return config

config = open_config(run, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")
ntrain = config['train_size']

# %% ---- Load model and samples ----
# modelpath = os.path.join(wd, 'results', 'mangroves', 'damagemodel.pkl')
# with open(modelpath, 'rb') as f:
#     model = load(f)
#     print(model.metrics)

samplespath = os.path.join(wd, 'samples', f'{run}.nc')
samples = xr.open_dataset(samplespath)
month_medians = samples.sel(month=month).medians
samples_month = samples[['anomaly', 'dependent', 'independent']] + month_medians
samples_month = samples_month.rename({'anomaly': 'hazGAN'})

datapath = os.path.join(wd, 'training', '18x22', 'data.nc')
data = xr.open_dataset(datapath).sel(channel=['u10', 'tp'])
data_month = data + month_medians # data['medians']
data_month = data_month.rename({'anomaly': 'era5', 'time': 'sample'})
train_month = data_month.isel(sample=slice(0, ntrain))
test_month = data_month.isel(sample=slice(ntrain, None))
yearly_rate = data.attrs['yearly_freq']

# for dev
if DEV:
    samples_month = samples_month.isel(sample=slice(0, 10))
    train_month = train_month.isel(sample=slice(0, 10))

# %% ---- Predict  ----
damages_sample = mangroves.get_damages(samples_month, ['dependent', 'independent', 'hazGAN'])
damages_train = mangroves.get_damages(train_month, ['era5'])
damages_test = mangroves.get_damages(test_month, ['era5'])

# %% ---- Plot mangrove damage predictions for random storm ----
if PLOT:
    i = np.random.randint(0, damages_sample.sizes['sample']-1)
    j = np.random.randint(0, damages_train.sizes['sample']-1)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    axs = axes[0, :]
    heatmap_kwargs = {'cmap':'YlOrRd', 'cbar_kwargs': {'label': 'Wind speed [mps]'}}
    damages_sample.isel(sample=i, channel=0).hazGAN.plot.contourf(ax=axs[0], **heatmap_kwargs)
    damages_train.isel(sample=j, channel=0).era5.plot.contourf(ax=axs[1], **heatmap_kwargs)
    axs[0].set_title(f'Wind speed [mps] (sample storm nᵒ{i})')
    axs[1].set_title(f'Wind speed [mps] (real storm nᵒ{j})')

    axs = axes[1, :]
    heatmap_kwargs = {'cmap':'PuBu', 'cbar_kwargs': {'label': 'Total precipitation [m]'}} 
    damages_sample.isel(sample=i, channel=1).hazGAN.plot.contourf(ax=axs[0], **heatmap_kwargs)
    damages_train.isel(sample=j, channel=1).era5.plot.contourf(ax=axs[1], **heatmap_kwargs)
    axs[0].set_title(f'Total precipitation [m] (sample storm nᵒ{i})')
    axs[1].set_title(f'Total precipitation [m] (real storm nᵒ{j})')

    axs = axes[2, :]
    heatmap_kwargs = {'cmap':'YlOrRd',
                    'cbar_kwargs': {
                        'label': 'Mangrove damage (%area)',
                        'format': matplotlib.ticker.PercentFormatter(1, 0)
                        }
                    }
    damages_sample.isel(sample=i).hazGAN_damage.plot.contourf(ax=axs[0], **heatmap_kwargs)
    damages_train.isel(sample=j).era5_damage.plot.contourf(ax=axs[1], **heatmap_kwargs)
    axs[0].set_title(f'Predicted mangrove damage (sample storm nᵒ{i})')
    axs[1].set_title(f'Predicted mangrove damage (real storm nᵒ{j})')

# %% ---- Step 2: Load mangrove data ----
aoi = box(xmin, ymin, xmax, ymax)
global_mangrove_watch = '/Users/alison/Documents/DPhil/data/gmw-v3-2020.nosync/gmw_v3_2020_vec.gpkg'
mangroves = gpd.read_file(global_mangrove_watch, mask=aoi)
mangroves = mangroves.set_crs(epsg=4326).drop(columns='PXLVAL')
mangroves['area']  = mangroves.to_crs(bay_of_bengal_crs).area
mangrove_centroids = mangroves.set_geometry(mangroves.centroid)
mangrove_centroids  = mangrove_centroids.sort_values(by='area', ascending=False)

mangrovesout = os.path.join(wd, 'results', 'mangroves', 'mangroves.geojson')
mangroves.to_file(mangrovesout, driver='GeoJSON')

# %% ---- Convert mangroves to xarray.Dataset with values as areas ----
mangroves_gridded = mangroves.intersect_with_damages(mangroves, damages_sample)
mangroves_gridded.area.plot(cmap="Greens", cbar_kwargs={'label': 'Mangrove damage [km²]'})

# %% ---- Calculate mangrove damage area and percentage ----
# for logistic regression probabilities this actually expected damage area
damages_sample['hazGAN_damagearea'] = mangroves_gridded.area * damages_sample.hazGAN_damage
damages_sample['hazGAN_damagepercent'] = damages_sample.hazGAN_damage.where(mangroves_gridded.area > 0)

damages_sample['dependent_damagearea'] = mangroves_gridded.area * damages_sample.dependent_damage
damages_sample['dependent_damagepercent'] = damages_sample.dependent_damage.where(mangroves_gridded.area > 0)

damages_sample['independent_damagearea'] = mangroves_gridded.area * damages_sample.independent_damage
damages_sample['independent_damagepercent'] = damages_sample.independent_damage.where(mangroves_gridded.area > 0)

damages_train['era5_damagearea'] = mangroves_gridded.area * damages_train.era5_damage
damages_train['era5_damagepercent'] = damages_train.era5_damage.where(mangroves_gridded.area > 0)

damages_test['era5_damagearea'] = mangroves_gridded.area * damages_test.era5_damage
damages_test['era5_damagepercent'] = damages_test.era5_damage.where(mangroves_gridded.area > 0)

# %% Turn contourf on and off
if PLOT:
    i = 7 # 969, i = np.random.randint(0, len(damages_sample.sample)
    damages = damages_sample.copy()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    mangroves_gridded.area.plot(ax=axs[0, 0], cmap="Greens",
                                cbar_kwargs={'label': 'Mangrove area [km²]'})
    damages.isel(sample=i).hazGAN_damage.plot(
        ax=axs[0, 1],
        cmap='YlOrRd',
        cbar_kwargs={'label': 'Probability of mangrove damage',
                    'format': PercentFormatter(1, 0)
                    }
                    )
    damages.isel(sample=i).hazGAN_damage.plot.contour(
        ax=axs[0, 1], colors='wheat', linewidths=.5)

    damages.isel(sample=i).hazGAN_damagepercent.plot(
        ax=axs[1, 0],
        cmap='YlOrRd',
        cbar_kwargs={'label': 'Probability of mangrove damage',
                    'format': PercentFormatter(1, 0)}
                    )

    damages.isel(sample=i).hazGAN_damagearea.plot(
        ax=axs[1, 1],
        cmap='YlOrRd',
        cbar_kwargs={'label': 'Expected area damaged [km²]'}
        )

    axs[0, 0].set_title('Mangrove area')
    axs[0, 1].set_title('Probability of mangrove damage')
    axs[1, 0].set_title('Probability of mangrove damage')
    axs[1, 1].set_title('Expected area damaged [km²]')

    for ax in axs.ravel():
        ax.add_feature(feature.COASTLINE)
        ax.add_feature(feature.BORDERS)
        ax.add_feature(feature.LAND, facecolor='wheat')
        ax.add_feature(feature.OCEAN)

    fig.suptitle('Sample storm damage to mangroves')

# %% ---- Calculate return periods and EADs ----
def auc_ufunc(x, y):
    x = sorted(x)
    y = sorted(y)
    out = auc(x, y)
    return out


def calculate_eads(var, damages: xr.Dataset, yearly_rate: int) -> xr.Dataset:
    """https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html"""
    basename = var.split('_')[0]
    damages = damages.copy()
    exceedence_prob = 1 - (damages[var].rank(dim='sample') / (1 + damages[var].sizes['sample']))

    annual_exceedence_prob = yearly_rate * exceedence_prob
    return_period = 1 / annual_exceedence_prob
    damages[f'{basename}_annual_exceedence_prob'] = annual_exceedence_prob
    damages[f'{basename}_return_period'] = return_period

    EADs = xr.apply_ufunc(
        auc_ufunc,
        damages[f'{basename}_annual_exceedence_prob'],
        damages[var],
        input_core_dims=[['sample'], ['sample']],
        output_core_dims=[[]],
        exclude_dims=set(('sample',)), # dimensions allowed to change size, must be set!
        vectorize=True,                # loop over non-core dimensions,
        dask="parallelized",
        output_dtypes=[float]
        )
    
    damages[f'{basename}_EAD'] = EADs
    return damages

damages_sample = calculate_eads('hazGAN_damagearea', damages_sample, occurrence_rate)
damages_sample = calculate_eads('independent_damagearea', damages_sample, occurrence_rate)
damages_sample = calculate_eads('dependent_damagearea', damages_sample, occurrence_rate)
damages_train = calculate_eads('era5_damagearea', damages_train, occurrence_rate)
damages_test = calculate_eads('era5_damagearea', damages_test, occurrence_rate)

# %% ---- Plot EADs ----
if PLOT:
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})

    damages_sample['dependent_EAD'].plot(ax=axs[0], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
    damages_sample['independent_EAD'].plot(ax=axs[1], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
    damages_sample['hazGAN_EAD'].plot(ax=axs[2], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
    damages_train['era5_EAD'].plot(ax=axs[3], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})

    axs[0].set_title('All dependent')
    axs[1].set_title('All independent')
    axs[2].set_title('hazGAN')
    axs[3].set_title('Real data')

# %% ---- Return period vs. damages plots (Lamb 2010) ----
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

totals_independent = calculate_total_return_periods(damages_sample, occurrence_rate, 'independent_damagearea')
totals_dependent = calculate_total_return_periods(damages_sample, occurrence_rate, 'dependent_damagearea')
totals_hazGAN = calculate_total_return_periods(damages_sample, occurrence_rate, 'hazGAN_damagearea')
totals_train = calculate_total_return_periods(damages_train, occurrence_rate, 'era5_damagearea')
totals_test = calculate_total_return_periods(damages_test, occurrence_rate, 'era5_damagearea')

# %% Save damages files for train and samples, as well as total damages
damages_sample_out = os.path.join(wd, 'results', 'mangroves', 'damages_sample.nc')
damages_train_out = os.path.join(wd, 'results', 'mangroves', 'damages_train.nc')
damages_test_out = os.path.join(wd, 'results', 'mangroves', 'damages_test.nc')
totals_out = os.path.join(wd, 'results', 'mangroves', 'totals.nc')

damages_sample.to_netcdf(damages_sample_out)
damages_train.to_netcdf(damages_train_out)
damages_test.to_netcdf(damages_test_out)

totals_independent.to_netcdf(totals_out, group='independent', mode='w')
totals_dependent.to_netcdf(totals_out, group='dependent', mode='a')
totals_hazGAN.to_netcdf(totals_out, group='hazGAN', mode='a')
totals_train.to_netcdf(totals_out, group='train', mode='a')
totals_test.to_netcdf(totals_out, group='test', mode='a')

# %% Lamb (2010) Figure 7, NOTE: new script just for this
if PLOT:
    fig, ax = plt.subplots()
    eps =.25
    mask = totals_hazGAN.return_period > eps
    ax.scatter(
        totals_hazGAN.where(mask).return_period,
        totals_hazGAN.where(mask).hazGAN_damagearea,
        color='k',
        linewidth=1.5,
        label='Modelled dependence'
    )
    mask = totals_dependent.return_period > eps
    ax.plot(
        totals_dependent.where(mask).return_period,
        totals_dependent.where(mask).dependent_damagearea,
        color='blue',
        linewidth=1.5,
        linestyle='dotted',
        label='Complete dependence'
    )
    mask = totals_independent.return_period > eps
    ax.plot(
        totals_independent.where(mask).return_period,
        totals_independent.where(mask).independent_damagearea,
        color='r',
        linestyle='dashed',
        linewidth=1.5,
        label='Independence'
    )
    mask = totals_train.return_period > eps
    ax.scatter(
        totals_train.where(mask).return_period,
        totals_train.where(mask).era5_damagearea,
        color='k',
        s=1.5,
        label='Training data'
    )
    ax.set_xlabel('Return period (years)')
    ax.set_ylabel('Total damage to mangroves (km²)')
    ax.legend()
    if not DEV:
        ax.set_xscale('log')
        ax.set_xticks([2, 5, 25, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter());
    # ax.set_title('Based on Figure 7 Lamb (2010)')

# %%
os.system('say done')
# %%
