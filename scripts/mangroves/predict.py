"""
Load a pretrained model then plot and display EADs
"""
# %% ---- Setup ----
import os
import numpy as np
from joblib import load
import xarray as xr
import matplotlib.pyplot as plt
plt.style.use('bmh')

month = 7
run = 'clean-sweep-3'
yearly_rate = 18 # from R
wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')

# %% ---- Load model and samples ----
modelpath = os.path.join(wd, 'results', 'mangroves', 'model.pkl')
with open(modelpath, 'rb') as f:
    model = load(f)

samplespath = os.path.join(wd, 'samples', f'{run}.nc')
samples = xr.open_dataset(samplespath)
month_medians = samples.median(dim='month') # OR: samples.sel(month=month)
samples_month = samples[['anomaly']] + month_medians
samples_month = samples_month.stack(grid=['lat', 'lon'])

# %% ---- Predict (needs Pandas)----
X = samples_month['anomaly'].to_dataframe()[['anomaly']]
X = X.unstack('channel').reset_index()
X.columns = ['sample', 'lat', 'lon', 'wind' ,'mslp']
X = X.set_index(['sample', 'lat', 'lon'])
X['mangrove_damage'] = model.predict(X)
damages = X['mangrove_damage'].to_xarray()

# %% ---- Plot mangrove damage predictions for random storm ----
fig, ax = plt.subplots()
i = np.random.random_integers(0, damages.sizes['sample'])
damages.isel(sample=i).plot(ax=ax, cmap="YlOrRd")
ax.set_title(f'Predicted mangrove damage (sample storm nᵒ {i})')

# %% ---- Calculate return periods and EADs ----
from sklearn.metrics import auc

def auc_ufunc(x, y):
    x = sorted(x)
    y = sorted(y)
    return auc(x, y)

exceedence_prob = 1 - (damages.rank(dim='sample') / (1 + damages.sizes['sample']))
return_period = 1 / (yearly_rate * exceedence_prob)
exceedence = 1 / (yearly_rate * return_period)

damages = damages.to_dataset()
damages['return_period'] = return_period
damages['exceedence'] = exceedence
damages['expected_annual'] = xr.apply_ufunc(auc_ufunc, damages['exceedence'], damages['mangrove_damage'], input_core_dims=[['sample'], ['sample']], vectorize=True)
damages.to_netcdf(os.path.join(wd, 'results', 'mangroves', 'damages.nc'))
# %%
import rioxarray as rio

tifpath = os.path.join(wd, 'results', 'mangroves', 'expected_annual_damages.tif')

damages['expected_annual'].rio.set_spatial_dims('lon', 'lat')\
    .rio.set_crs(4326)\
    .rio.to_raster(tifpath, driver='GTiff')

# %% ---- Load mangrove data ----
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

mangrovesout = os.path.join(wd, 'results', 'mangroves', 'mangroves.geojson')
mangroves.to_file(mangrovesout, driver='GeoJSON')

# %% ---- Overlay with EADs ----
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
damages['expected_annual'].plot(
    ax=ax,
    cmap='YlOrRd',
    cbar_kwargs={
        'label': 'Expected annual damage [\% area]',
        'format': PercentFormatter(1, 0)
        })
mangrove_centroids.plot(ax=ax,
                         color='lightgreen',
                         alpha=.8,
                         edgecolor='k',
                         linewidth=0.5,
                         markersize=mangroves.area * 5000
                         )
ax.set_title('Expected annual damages for mangroves\nin the Bay of Bengal');
# %% ---- (1) Get EADs for individual mangrove patches ----
import xagg as xa

weightmap = xa.pixel_overlaps(damages, mangroves)
aggregated = xa.aggregate(damages[['expected_annual']], weightmap)
mangrove_damages = aggregated.to_geodataframe()
mangrove_damages.columns = ['area', 'EAPD']
mangrove_damages = gpd.GeoDataFrame(mangrove_damages, geometry=mangrove_centroids.geometry)
mangrove_damages = mangrove_damages.set_crs(4326)
mangrove_damages['EAD'] = mangrove_damages['area'] * mangrove_damages['EAPD']

# %%
mangrove_damages = mangrove_damages.sort_values(by='EAD', ascending=True)
mangrove_damages.plot(
    'EAD',
    cmap='YlOrRd',
    edgecolor='k',
    linewidth=.1,
    legend=True,
    alpha=.8,
    legend_kwds={'label': 'Expected annual damage [m²]'}
    )

# %% ---- Transform to a raster for nicer plotting ----
# https://pygis.io/docs/e_raster_rasterize.html
from rasterio import features
from rasterio.plot import show
from rasterio.enums import MergeAlg
import rasterio
from matplotlib.colors import Normalize, LogNorm

lats = np.linspace(ymax, ymin, src.shape[0])[::-1]
lons = np.linspace(xmin, xmax, src.shape[1])

geom = [*mangroves.geometry]
geom_value = ((geom, value) for geom, value in zip(geom, mangrove_damages['EAD']))
src = rasterio.open(tifpath)

rasterized = features.rasterize(geom_value,
                                out_shape=src.shape,
                                fill=0,
                                out=None,
                                transform=src.transform,
                                all_touched=True,
                                merge_alg=MergeAlg.add,
                                dtype=np.float32)

ds = xr.Dataset(
    {
        'expected_annual_damages': (
            ['lat', 'lon'],
            rasterized,
            {
                'long_name': 'Expected Annual Damages',
                'units': 'm²',
                'display_name': 'Expected Annual Damages (m²)'
            }
        )
    },
    coords={
        'lat': ('lat', lats, {'long_name': 'Latitude', 'units': 'degrees_north'}),
        'lon': ('lon', lons, {'long_name': 'Longitude', 'units': 'degrees_east'})
    },
    attrs={
        'crs': src.crs,
        'transform': src.transform,
        'area_units': 'm²'
    }
)



ds.expected_annual_damages.plot(cmap='YlOrRd', norm=LogNorm(), vmin=1e5)
# %%
