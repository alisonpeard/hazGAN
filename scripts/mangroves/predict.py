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

global model

month = 7
DEV = False
SMOOTHING = False
run = 'vital-sweep-30__precipsota'
yearly_rate = 18 # from R
wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')

# %% ---- Load model and samples ----
modelpath = os.path.join(wd, 'results', 'mangroves', 'damagemodel.pkl')
scalerpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'scaler.pkl')
with open(modelpath, 'rb') as f:
    model = load(f)
    print(model.metrics)

samplespath = os.path.join(wd, 'samples', f'{run}.nc')
samples = xr.open_dataset(samplespath)
month_medians = samples.sel(month=month).medians
samples_month = samples[['anomaly', 'dependent', 'independent']] + month_medians
samples_month = samples_month.rename({'anomaly': 'hazGAN'})
# samples_month = samples_month.stack(grid=['lat', 'lon'])

datapath = os.path.join(wd, 'training', '18x22', 'data.nc')
data = xr.open_dataset(datapath)
data_month = data + month_medians # data['medians']
data_month = data_month.rename({'anomaly': 'era5', 'time': 'sample'})
# data_month = data_month.stack(grid=['lat', 'lon'])
train_month = data_month.isel(sample=slice(0, 560))
test_month = data_month.isel(sample=slice(560, None))

# for dev
if DEV:
    samples_month = samples_month.isel(sample=slice(0, 10))
    train_month = train_month.isel(sample=slice(0, 10))
# %% ---- Predict  ----
def damage_ufunc(X):
    X = X.reshape(1, -1)
    return model.predict(X)

def get_damages(
    ds: xr.DataArray,
    vars_: list,
    first_dim='sample',
    core_dim='channel',
    ) -> xr.DataArray:
    """
    Calculate mangrove damages using a pretrained model.
    
    https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
    """
    predictions = ds.copy()
    ds = ds.copy().chunk({first_dim: "auto"})
    for var in vars_:
        predictions[f"{var}_damage"] = xr.apply_ufunc(
            damage_ufunc,
            ds[var],
            input_core_dims=[[core_dim]],  # apply ufunc along
            output_core_dims=[[]],         # dimensions results are in
            exclude_dims=set((core_dim,)), # dimensions allowed to change size, must be set!
            vectorize=True,                # loop over non-core dimensions
            dask="parallelized",
            output_dtypes=[float]
        )
    return predictions


damages_sample = get_damages(samples_month, ['dependent', 'independent', 'hazGAN'])
damages_train = get_damages(train_month, ['era5'])

# %% ---- Plot mangrove damage predictions for random storm ----
i = np.random.randint(0, damages_sample.sizes['sample']-1)
j = np.random.randint(0, damages_train.sizes['sample']-1)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

axs = axes[0, :]
heatmap_kwargs = {'cmap':'YlOrRd', 'cbar_kwargs': {'label': 'Wind speed [mps]'}}
damages_sample.isel(sample=i, channel=0).hazGAN.plot(ax=axs[0], **heatmap_kwargs)
damages_train.isel(sample=j, channel=0).era5.plot(ax=axs[1], **heatmap_kwargs)
axs[0].set_title(f'Wind speed [mps] (sample storm nᵒ{i})')
axs[1].set_title(f'Wind speed [mps] (real storm nᵒ{j})')

axs = axes[1, :]
heatmap_kwargs = {'cmap':'PuBu', 'cbar_kwargs': {'label': 'Total precipitation [m]'}} 
damages_sample.isel(sample=i, channel=1).hazGAN.plot(ax=axs[0], **heatmap_kwargs)
damages_train.isel(sample=j, channel=1).era5.plot(ax=axs[1], **heatmap_kwargs)
axs[0].set_title(f'Total precipitation [m] (sample storm nᵒ{i})')
axs[1].set_title(f'Total precipitation [m] (real storm nᵒ{j})')

axs = axes[2, :]
heatmap_kwargs = {'cmap':'YlOrRd',
                  'cbar_kwargs': {
                      'label': 'Mangrove damage (%area)',
                      'format': matplotlib.ticker.PercentFormatter(1, 0)
                      }
                  }
damages_sample.isel(sample=i).hazGAN_damage.plot(ax=axs[0], **heatmap_kwargs)
damages_train.isel(sample=j).era5_damage.plot(ax=axs[1], **heatmap_kwargs)
axs[0].set_title(f'Predicted mangrove damage (sample storm nᵒ{i})')
axs[1].set_title(f'Predicted mangrove damage (real storm nᵒ{j})')

# %% PAUSE HERE UNTIL FIGURES LOOK RIGHT


















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
    # damages = damages_sample.copy()
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

mangroves_gridded = intersect_mangroves_with_damages(mangroves, damages_sample)
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

# %% Explore smoothing
if SMOOTHING: 
    # %% just an example to incorporate later for paper figures
    # NOTE: I think interpolation might be better than smoothing
    from hazGAN import gaussian_blur

    ds = damages_sample.copy()#.isel(sample=0)
    da = ds.hazGAN_damage

    x = da.data[..., np.newaxis]
    xblur = gaussian_blur(x, kernel_size=2, sigma=5).numpy().squeeze()
    ds['smoothed'] = (['sample', 'lat', 'lon'], xblur)

    i = np.random.randint(0, damages_sample.sizes['sample'])
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ds.isel(sample=i).hazGAN_damage.plot(ax=ax[0, 0], cmap='YlOrRd')
    ds.isel(sample=i).smoothed.plot(ax=ax[0, 1], cmap='YlOrRd')
    ds.isel(sample=i).hazGAN_damage.plot.contourf(ax=ax[1, 0], cmap='YlOrRd', levels=10)
    ds.isel(sample=i).smoothed.plot.contourf(ax=ax[1, 1], cmap='YlOrRd', levels=10)
    fig.suptitle(f'Smoothing for sample {i}')

# %% Turn contourf on and off
import cartopy.crs as ccrs
from cartopy import feature
from matplotlib.ticker import PercentFormatter

i = 7# 969
damages = damages_sample.copy()
# i = np.random.randint(0, len(damages_sample.sample)


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
from sklearn.metrics import auc
from hazGAN import occurrence_rate

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
        vectorize=True, # loop over non-core dimensions,
        dask="parallelized",
        output_dtypes=[float]
        )
    
    damages[f'{basename}_EAD'] = EADs
    return damages

damages_sample = calculate_eads('hazGAN_damagearea', damages_sample, occurrence_rate)
damages_sample = calculate_eads('independent_damagearea', damages_sample, occurrence_rate)
damages_sample = calculate_eads('dependent_damagearea', damages_sample, occurrence_rate)
damages_train= calculate_eads('era5_damagearea', damages_train, occurrence_rate)


# %% ---- Plot EADs ----
fig, axs = plt.subplots(1, 4, figsize=(15, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})

damages_sample['dependent_EAD'].plot(ax=axs[0], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_sample['independent_EAD'].plot(ax=axs[1], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_sample['hazGAN_EAD'].plot(ax=axs[2], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})
damages_train['era5_EAD'].plot(ax=axs[3], cmap='YlOrRd', cbar_kwargs={'label': 'Expected annual damages [km²]'})

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

totals_independent = calculate_total_return_periods(damages_sample, occurrence_rate, 'independent_damagearea')
totals_dependent = calculate_total_return_periods(damages_sample, occurrence_rate, 'dependent_damagearea')
totals_hazGAN = calculate_total_return_periods(damages_sample, occurrence_rate, 'hazGAN_damagearea')
totals_data = calculate_total_return_periods(damages_train, occurrence_rate, 'era5_damagearea')

# %%
fig, ax = plt.subplots()
eps =.25

mask = totals_hazGAN.return_period > eps
ax.plot(
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

mask = totals_data.return_period > eps
ax.scatter(
    totals_data.where(mask).return_period,
    totals_data.where(mask).era5_damagearea,
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
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title('Based on Figure 7 Lamb (2010)')
# %%
!say done
# %%
