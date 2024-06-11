"""Load observations/synthetic data and calculate mangrove damage probabilities.

Visualise points: https://www.researchgate.net/publication/271193539_Oil_Spills_in_Mangroves_Planning_Response_Considerations/figures?lo=1
"""
# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import xagg as xa
import hazGAN as hg
import geopandas as gpd
from sklearn.metrics import auc
from shapely.geometry import box
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cartopy.crs as ccrs
plt.style.use('bmh')

wd = os.path.join(os.getcwd(), "..", '..')  # hazGAN directory
plot_kwargs = {"bbox_inches": "tight", "dpi": 300}
# %% load the mangrove shapefile from Global Mangrove Watch (downloaded 27/05/2024)
mangrove_path = '/Users/alison/Documents/DPhil/data/global_mangrove_watch/v3_2020/gmw_v3_2020_vec.gpkg'
mangroves =  gpd.read_file(mangrove_path, bbox=(80, 10, 95, 25))
# %% load the ERA5 data or generated data
datadir = os.path.join(wd, "..", "era5_data.nosync")  # keep data folder in parent directory
data = xr.open_dataset(os.path.join(datadir, "data.nc")) 
# %%
# make a new dataset with full scale winds and pressure
m = len(pd.unique(data['time']))
n = len(pd.unique(data['time.year']))
yearly_rate = m / n # lambda

X = data['anomaly'].data + data['medians'].data
n, h, w, c = X.shape
lats = data['lat'].data
lons = data['lon'].data

ds = xr.Dataset(
    {
        "wind_speed": (("time", "latitude", "longitude"), X[..., 0], {"units": "mps"}),
        "mslp": (("time", "latitude", "longitude"), X[..., 1], {"units": "Pa"}),
    },
    coords={
        "time": data['time'],
        #"event": np.arange(n),
        "latitude": lats,
        "longitude": lons,
    },
)
# %% Define mangrove damage function ##########################################################################
mangrove_df = pd.read_csv("/Users/alison/Documents/DPhil/multivariate/mangrove_data/input.csv")
mangrove_df['Wind_landfall'] = mangrove_df['Wind_landfall'] * 1000 / 3600  # convert to mps
mangrove_df['damage_prob'] = 0
mangrove_df.loc[mangrove_df['Damage'] > 0, 'damage_prob'] = 1
model = sm.GLM(mangrove_df['damage_prob'].values, sm.add_constant(mangrove_df['Wind_landfall'].values), family=sm.families.Binomial())
results = model.fit()
results.summary()
# %%
mangrove_df = mangrove_df.sort_values('HurricaneYear')
plt.scatter(mangrove_df['Wind_landfall'], mangrove_df['Damage'], color='k', s=10)
plt.xlabel('Wind speed at landfall (m/s)')
plt.ylabel('Mangrove damage')

# %%
pred = pd.DataFrame({"Wind_landfall": np.arange(0, 250 * 1000 / 3600, 1)})
pred['fit'] = results.predict(sm.add_constant(pred['Wind_landfall'].values), which='linear')
pred["fit2"] = results.predict(sm.add_constant(pred['Wind_landfall'].values), which='mean')

# plt.scatter(mangrove_df['Wind_landfall'], mangrove_df['damage_prob'],linewidth=0.2, marker='o', edgecolor='k', color='white')
plt.plot(pred['Wind_landfall'], pred['fit2'], color='blue', linewidth=2)
plt.xlabel('Wind speed at landfall (m/s)')
plt.ylabel('Probability of mangrove damage')
plt.title('Mangrove damage probability curve (global)')
# %% 
def damage_function(x, res):
    def func(x):
        print(x.shape)
        shape = x.shape
        y = res.predict(sm.add_constant(x.ravel()), which='mean')
        y = y.reshape(shape)
        print(y.shape)
        return y
    return xr.apply_ufunc(func, x, vectorize=False, input_core_dims=[['time', 'latitude', 'longitude']], output_core_dims=[['time', 'latitude', 'longitude']])

ds['damage_prob'] = damage_function(ds.wind_speed, results)
ds['longitude'] = np.linspace(80, 95, 21)
ds['latitude'] = np.linspace(10, 25, 21)

# %%
EVENT = np.random.randint(0, 1606) # 20
# TIME = ds.where(ds.damage_prob == ds.damage_prob.max(), drop=True).time.values[0]
fig = plt.figure(figsize=(12, 4), layout='tight')
ax  = plt.axes((0, 0, 0.5, 0.9), projection =ccrs.PlateCarree()) 
ax.coastlines(resolution='50m',color='k', linewidth=.5) 
ds['wind_speed'].isel(time=EVENT).plot.contourf(levels=20, cmap='viridis', ax=ax, cbar_kwargs={'label': 'Wind speed [mps]'})
# plt.title('Wind speed for event 20')

ax = plt.axes((0.5, 0, 0.5, 0.9), projection=ccrs.PlateCarree()) 
ax.coastlines(resolution='50m',color='k', linewidth=.5) 
ds['damage_prob'].isel(time=EVENT).plot.contourf(levels=20, cmap='YlOrRd', ax=ax, cbar_kwargs={'label': 'Mangrove damage probability', 'format': PercentFormatter(1, 0)})
# %% Combine with mangrove data for 2020 ##########################################################################
ds = ds.drop_vars(['wind_speed', 'mslp'])
weightmap = xa.pixel_overlaps(ds, mangroves)
aggregated = xa.aggregate(ds, weightmap)
# %% mke a geodataframe for visualisation
df_agg = aggregated.to_dataframe()
df_agg = df_agg.drop(columns='PXLVAL')
mangroves = mangroves.drop(columns='PXLVAL')
df_agg = df_agg.join(mangroves, on='poly_idx', how='left')
gdf_agg = gpd.GeoDataFrame(df_agg, geometry='geometry')
# %% Spatially explicit EADs ##########################################################################
def get_return_period(x: pd.Series, lam=yearly_rate):
    # #https://georgebv.github.io/pyextremes/user-guide/6-return-periods/ 
    n = len(x)
    r = x.rank(ascending=False, method='first')
    p = r / (n + 1)
    rp = 1 / p / lam
    return rp

idx = pd.IndexSlice
pixels = [*gdf_agg.reset_index()['poly_idx'].unique()]
gdf_agg['return_period'] = [np.nan] * len(gdf_agg)
for i in pixels:
    gdf_agg.loc[idx[i, :], 'return_period'] = get_return_period(gdf_agg.loc[idx[i, :], 'damage_prob'])

# %% start with a single polygon-damage probability curve
i = np.random.randint(0, len(pixels))
poly = gdf_agg.loc[idx[i, :]]
poly['exceedence'] = 1 / (yearly_rate * poly['return_period'])
poly = poly.sort_values('exceedence', ascending=False)

fig, ax = plt.subplots()
ax.plot(poly['return_period'], poly['damage_prob'])
ax.set_title(f'Damage probability curve for pixel {i}')
ax.fill_between(poly['return_period'], poly['damage_prob'], alpha=0.5)
ax.set_xlim(ax.get_xlim()[::-1])
ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
ax.set_xlabel('Return period (years)')
ax.set_ylabel('Damage probability')

# area under the curve (i.e. EAD)
eapd = auc(poly['exceedence'], poly['damage_prob'])
print(f'EAD for pixel {i}: {eapd}')
# %% now do it for all pixels
def get_ead(x: pd.DataFrame):
    x = x.sort_values('exceedence', ascending=False)
    return auc(x['exceedence'], x['damage_prob'])

gdf_agg['exceedence'] = 1 / (yearly_rate * gdf_agg['return_period'])
gdf_ead = gdf_agg.groupby('poly_idx')
gdf_ead = gdf_ead.apply(get_ead).reset_index()
gdf_ead = gdf_ead.rename(columns={0: 'EAD'})
gdf_ead = gdf_ead.join(mangroves[['geometry']], on='poly_idx', how='left')
gdf_ead = gpd.GeoDataFrame(gdf_ead, geometry='geometry').set_crs(4326)

zoom = box(93, 19, 94, 20)
zoom_gdf = gpd.GeoDataFrame(geometry=[zoom]).set_crs(4326)
# %% need to make this better looking
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax = axs[0]
gdf_ead.plot('EAD', cmap='YlOrRd', legend=True, legend_kwds={'label':'EAD', 'format': PercentFormatter(1, 0)}, ax=axs[0])
zoom_gdf.boundary.plot(color='k', ax=ax, linewidth=.5)
ax.set_title('EAD (probability)')

ax = axs[1]
gdf_ead.clip(zoom).plot('EAD', cmap='YlOrRd', legend=True, ax=ax, legend_kwds={'label': 'EAD', 'format': PercentFormatter(1, 0)})
ax.set_title('EAD (zoomed in)')

# %% Let's try it as an xarray dataset instead
ds_agg = xr.Dataset.from_dataframe(gdf_agg, sparse=True)


# %% SPATIALLY AGGREGATED ##########################################################################
gdf_bob = gdf_agg.groupby('event').agg({'damage_prob': 'mean'}).reset_index()
gdf_bob['return_period'] = get_return_period(gdf_bob['damage_prob'])
gdf_bob = gdf_bob.sort_values('return_period', ascending=True)

fig, ax = plt.subplots()
ax.scatter(gdf_bob['return_period'], gdf_bob['damage_prob'], s=1, color='k')
ax.set_xlabel('Return period [years]')
ax.set_ylabel('Average damage probabillty')
ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
ax.set_title('Bay of Bengal mangrove damages from wind')

# %%
