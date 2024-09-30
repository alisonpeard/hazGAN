"""
TODO: Fix, why is max RP for samples 10???
Script to predict mangrove damage from storm data using a trained hazGAN model.

Input files:
-------
- config-defaults.yaml: configuration file
- results/mangroves/damagemodel.pkl: trained model
- samples/amber-sweep-13.nc: generated storm footprints
- training/18x22/data.nc: ERA5 storm foorprints
- data/gmw-v3-2020_vec.gpkg: global mangrove data

Output files:
--------
- results/mangroves/mangroves.geojson: mangroves clipped to the area of interest
- results/mangroves/mangroves_grid.nc: gridded mangroves for damage calculations
- results/mangroves/damages_sample.nc: predicted damages for each storm sample
- results/mangroves/damages_train.nc: predicted damages for each training storm
- results/mangroves/damages_test.nc: predicted damages for each test storm
- results/mangroves/totals.nc: total predicted damages for each storm type
"""
# %% user-defined arguments for script
MONTH   = 7
PLOT    = False
DEV     = True
RUN     = 'amber-sweep-13'
WD      = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')
DATADIR = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'data')

# %% environment
import os
import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from joblib import load
import xarray as xr
import xagg as xa
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cartopy.crs as ccrs
from cartopy import feature
from sklearn.metrics import auc
import warnings

from hazGAN import (
    xmin, xmax, ymin, ymax,
    bay_of_bengal_crs
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']
plt.style.use('bmh')

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    return {key: value["value"] for key, value in config.items()}

def load_model_and_scaler(modelpath, scalerpath):
    with open(modelpath, 'rb') as f:
        model = load(f)
    return model

def load_data(samplespath, datapath, month, ntrain):
    samples = xr.open_dataset(samplespath)
    month_medians = samples.sel(month=month).medians
    samples_month = samples[['anomaly', 'dependent', 'independent']] + month_medians
    samples_month = samples_month.rename({'anomaly': 'hazGAN'})

    data = xr.open_dataset(datapath).sel(channel=['u10', 'tp'])
    data_month = data + month_medians 
    data_month = data_month.rename({'anomaly': 'era5', 'time': 'sample'})
    train_month = data_month.isel(sample=slice(0, ntrain))
    test_month = data_month.isel(sample=slice(ntrain, None))
    occurrence_rate = data.attrs['yearly_freq']
    
    # use chunking to improve efficiency
    samples_month = samples_month.chunk({'sample': 100})
    train_month = train_month.chunk({'sample': 100})
    test_month = test_month.chunk({'sample': 100})

    # make sure using parallelized dask
    samples_month = samples_month.persist()
    train_month = train_month.persist()
    test_month = test_month.persist()
    

    return samples_month, train_month, test_month, occurrence_rate


def damage_ufunc(X):
    X = X.reshape(1, -1)
    return model.predict(X)


def auc_ufunc(x, y):
    x = sorted(x)
    y = sorted(y)
    out = auc(x, y)
    return out


def get_damages(ds, vars_, first_dim='sample', core_dim='channel'):
    predictions = ds.copy()
    ds = ds.copy().chunk({first_dim: "auto"})
    for var in vars_:
        predictions[f"{var}_damage"] = xr.apply_ufunc(
            damage_ufunc,
            ds[var],
            input_core_dims=[[core_dim]],
            output_core_dims=[[]],
            exclude_dims=set((core_dim,)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float]
        )
    return predictions


def intersect_mangroves_with_damages(mangroves, damages, plot=False):
    mangroves = mangroves.to_crs(4326)
    weightmap = xa.pixel_overlaps(damages, mangroves)

    mangroves_gridded = weightmap.agg
    mangroves_gridded['npix'] = mangroves_gridded['pix_idxs'].apply(len)
    mangroves_gridded['rel_area'] = mangroves_gridded['rel_area'].apply(lambda x: np.squeeze(x, axis=0))
    mangroves_gridded = mangroves_gridded.explode(['rel_area', 'pix_idxs'])

    mangroves_gridded['area'] = mangroves_gridded['area'] * mangroves_gridded['rel_area']
    mangroves_gridded = mangroves_gridded.groupby('pix_idxs').agg({'area': 'sum', 'coords': 'first'})

    lons = weightmap.source_grid['lon'].values
    lats = weightmap.source_grid['lat'].values
    mangroves_gridded = mangroves_gridded.reset_index()
    mangroves_gridded['lon'] = mangroves_gridded['pix_idxs'].apply(lambda j: lons[j])
    mangroves_gridded['lat'] = mangroves_gridded['pix_idxs'].apply(lambda i: lats[i])
    mangroves_gridded['area'] = mangroves_gridded['area'].astype(float) * 1e-6
    mangroves_gridded = mangroves_gridded.set_index(['lat', 'lon'])[['area']]
    mangroves_gridded = xr.Dataset.from_dataframe(mangroves_gridded)

    if plot:
        mangroves_gridded.area.plot(cmap="Greens", cbar_kwargs={'label': 'Mangrove damage [km²]'})
    return mangroves_gridded


def calculate_damage_areas(damages, mangroves_gridded):
    for damage_type in ['era5', 'hazGAN', 'dependent', 'independent']:
        if f'{damage_type}_damage' in damages:
            damages[f'{damage_type}_damagearea'] = mangroves_gridded.area * damages[f'{damage_type}_damage']
            damages[f'{damage_type}_damagepercent'] = damages[f'{damage_type}_damage'].where(mangroves_gridded.area > 0)
    return damages


def plot_mangrove_damage_predictions(damages_sample, damages_train, mangroves_gridded, PLOT=False):
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
                            'format': PercentFormatter(1, 0)
                            }
                        }
        damages_sample.isel(sample=i).hazGAN_damage.plot.contourf(ax=axs[0], **heatmap_kwargs)
        damages_train.isel(sample=j).era5_damage.plot.contourf(ax=axs[1], **heatmap_kwargs)
        axs[0].set_title(f'Predicted mangrove damage (sample storm nᵒ{i})')
        axs[1].set_title(f'Predicted mangrove damage (real storm nᵒ{j})')


def calculate_eads(var, damages, yearly_rate):
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
        exclude_dims=set(('sample',)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    damages[f'{basename}_EAD'] = EADs
    return damages


def calculate_total_return_periods(damages, yearly_rate, var='mangrove_damage_area'):
    totals = damages[var].sum(dim=['lat', 'lon']).to_dataset()
    N = totals[var].sizes['sample']
    totals['rank'] = totals[var].rank(dim='sample')
    totals['exceedence_probability'] = 1 - ( totals['rank'] / ( N + 1 ) )
    totals['return_period'] = 1 / ( yearly_rate * totals['exceedence_probability'] )
    totals = totals.sortby('return_period')
    return totals


def save_damages(damages_sample, damages_train, damages_test, totals_independent, totals_dependent, totals_hazGAN, totals_train, totals_test, wd):
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


# %% Main script
if __name__ == "__main__":
    config = open_config(RUN, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")
    ntrain = config['train_size']

    modelpath = os.path.join(WD, 'results', 'mangroves', 'damagemodel.pkl')
    scalerpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'scaler.pkl')
    model = load_model_and_scaler(modelpath, scalerpath)

    samplespath = os.path.join(WD, 'samples', f'{RUN}.nc')
    datapath = os.path.join(WD, 'training', '18x22', 'data.nc')
    samples_month, train_month, test_month, occurrence_rate = load_data(samplespath, datapath, MONTH, ntrain)

    # %%
    damages_sample = get_damages(samples_month, ['dependent', 'independent', 'hazGAN'])
    damages_train = get_damages(train_month, ['era5'])
    damages_test = get_damages(test_month, ['era5'])

    # x = damages_sample.hazGAN_damage.values
    if PLOT:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        damages_train.era5_damage.plot.hist(ax=axs[0], bins=50, alpha=0.5, label='train')
        damages_test.era5_damage.plot.hist(ax=axs[1], bins=50, alpha=0.5, label='test')
        damages_sample.hazGAN_damage.plot.hist(ax=axs[2], bins=50, alpha=0.5, label='sample')

    #%% clip mangroves to aoi
    mangrovesout = os.path.join(WD, 'results', 'mangroves', 'mangroves.geojson')
    if not os.path.exists(mangrovesout):
        print('Clipping global mangroves...')
        aoi = box(xmin, ymin, xmax, ymax)
        global_mangrove_watch = os.path.join(DATADIR, 'gmw-v3-2020.nosync', 'gmw_v3_2020_vec.gpkg')
        mangroves = gpd.read_file(global_mangrove_watch, mask=aoi)
        mangroves = mangroves.set_crs(epsg=4326).drop(columns='PXLVAL')
        mangroves['area'] = mangroves.to_crs(bay_of_bengal_crs).area
        mangroves.to_file(mangrovesout, driver='GeoJSON')
    else:
        print('Loading mangroves...')
        mangroves = gpd.read_file(mangrovesout)

    # %% grid mangroves to match damage resolution
    mangrovegrid_out = os.path.join(WD, 'results', 'mangroves', 'mangroves_grid.nc')
    if not os.path.exists(mangrovegrid_out):
        print('Gridding mangroves...')
        mangroves_gridded = intersect_mangroves_with_damages(mangroves, damages_sample)
        mangroves_gridded.to_netcdf(mangrovegrid_out)
    else:
        print('Loading mangroves grid...')
        mangroves_gridded = xr.open_dataset(mangrovegrid_out)

    # %% calculate damages per-storm
    print('Calculating damage areas...')
    damages_sample = calculate_damage_areas(damages_sample, mangroves_gridded)
    damages_train = calculate_damage_areas(damages_train, mangroves_gridded)
    damages_test = calculate_damage_areas(damages_test, mangroves_gridded)

    plot_mangrove_damage_predictions(damages_sample, damages_train, mangroves_gridded, PLOT)
    if PLOT:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        damages_sample.hazGAN_damagearea.plot.hist(ax=axs[0], bins=50, alpha=0.5, label='sample')
        damages_test.era5_damagearea.plot.hist(ax=axs[1], bins=50, alpha=0.5, label='test')
        damages_train.era5_damagearea.plot.hist(ax=axs[2], bins=50, alpha=0.5, label='train')


    # %% calculate EADs
    print('Calculating EADs...')
    damages_sample = calculate_eads('hazGAN_damagearea', damages_sample, occurrence_rate)
    damages_sample = calculate_eads('independent_damagearea', damages_sample, occurrence_rate)
    damages_sample = calculate_eads('dependent_damagearea', damages_sample, occurrence_rate)
    damages_train = calculate_eads('era5_damagearea', damages_train, occurrence_rate)
    damages_test = calculate_eads('era5_damagearea', damages_test, occurrence_rate)

    # %% calculate total return periods
    print('Calculating total return periods...')
    totals_independent = calculate_total_return_periods(damages_sample,occurrence_rate, 'independent_damagearea')
    totals_dependent = calculate_total_return_periods(damages_sample, occurrence_rate, 'dependent_damagearea')
    totals_hazGAN = calculate_total_return_periods(damages_sample, occurrence_rate, 'hazGAN_damagearea')
    totals_train = calculate_total_return_periods(damages_train, occurrence_rate, 'era5_damagearea')
    totals_test = calculate_total_return_periods(damages_test, occurrence_rate, 'era5_damagearea')
    print(f'Maximum return period: {totals_hazGAN.return_period.max().values}')

    if PLOT:
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        totals_independent.return_period.plot.hist(ax=axs[0], yscale='log')
        totals_dependent.return_period.plot.hist(ax=axs[1], yscale='log')
        totals_hazGAN.return_period.plot.hist(ax=axs[2], yscale='log')
        totals_train.return_period.plot.hist(ax=axs[3], yscale='log')
        totals_test.return_period.plot.hist(ax=axs[4], yscale='log')

    # %% save results
    print('Saving results...')
    save_damages(
        damages_sample, damages_train, damages_test, totals_independent,
        totals_dependent, totals_hazGAN, totals_train, totals_test, WD
        )
    os.system('say done')
    print('Done!')
    # %%
# %%
