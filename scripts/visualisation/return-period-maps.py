"""
Make wind speed/mslp maps for different return periods and month of the year using training data.
"""
#%%
import os
import numpy as np
import xarray as xr
import hazGAN as hg
import calendar
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

datadir = '/Users/alison/Documents/DPhil/multivariate/era5_data.nosync'
data = xr.open_dataset(os.path.join(datadir, "hazGAN_samples.nc"))
occurence_rate = 18.033 # from R 
sample_var = "sample"
# %%
vals = data.anomaly.values
vals = hg.gaussian_blur(vals, kernel_size=2, sigma=0.8)
data['anomaly'] = ([sample_var, 'lat', 'lon', 'channel'], vals)
if not 'month' in [*data.coords]:
    data['month'] = data['time'].dt.month
# %% 
if False:
    max_winds = data.isel(channel=0).anomaly.max(dim=['lat', 'lon'])
    m = max_winds.sample.size
    exceed_prob = 1 - (max_winds.rank(dim=sample_var) / (m + 1))
    rp = 1 / (occurence_rate * exceed_prob)
    data['storm_rp'] = ([sample_var], rp.values)

# %% calculate return periods across each pixel
wind = data.anomaly.sel(channel='u10')
m = wind[sample_var].size
exceed_prob = 1 - (wind.rank(dim=sample_var) / (m + 1))
rp = 1 / (occurence_rate * exceed_prob)
wind['return_period'] = rp
wind['aep'] = 1 / rp
# %% interpolate return periods
def get_rp_wind(x, y, rp):
    interpolated = np.interp(rp, np.sort(x), np.sort(y)) # TODO: interpolate better
    return interpolated
# %% ----Visualise RP wind speeds for a given month----
import matplotlib as mpl
plt.rcParams["font.family"] = "serif"

month = 6
median = data['median'].where(data['month']==month, drop=True).isel(channel=0)[0]
vmin = 15
vmax = 70
cmap = mpl.cm.Spectral_r
cmap.set_under('white')
cmap.set_over('darkred')

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, axs = plt.subplots(1, 4, figsize=(18, 4), subplot_kw={'projection': ccrs.PlateCarree()})
for ax, RP in zip(axs, [2, 5, 20, 40]):
    res = xr.apply_ufunc(get_rp_wind, rp, wind,
                        input_core_dims=[[sample_var], [sample_var]],
                        dask = 'allowed',
                        kwargs={'rp': RP},
                        vectorize = True)

    ax.coastlines(resolution='50m',color='k', linewidth=.5) 
    im = (median + res).plot(cmap=cmap, ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)
    (median + res).plot.contourf(levels=15, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False) #

    ax.set_title(f'{RP}-year storm')
plt.suptitle(f'Wind speeds for {calendar.month_name[month]} storms', y=1, fontsize='x-large')
plt.colorbar(im, ax=axs, extend='both', orientation='horizontal', label='Wind speed [m/s]',
             aspect=80, shrink=0.8)
# %% ----Compare this to independent RP maps-----
from scipy.stats import genpareto
month = 10
median = data['median'].where(data['month']==month, drop=True).isel(channel=0)[0]

def get_wind(params, RP):
    return genpareto.ppf(1-1/RP, *params)

fig, axs = plt.subplots(1, 4, figsize=(18, 4), subplot_kw={'projection': ccrs.PlateCarree()})
RPs = [5, 25, 50, 100]
for ax, RP in zip(axs, RPs):
    res = xr.apply_ufunc(get_wind, data['params'],
                            input_core_dims = [['param']],
                            dask = 'allowed',
                            kwargs={'RP': RP * occurence_rate},
                            vectorize = True)


    ax.coastlines(resolution='50m',color='k', linewidth=.5) 
    im = (median + res).isel(channel=0).plot(cmap=cmap, ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)
    (median + res).isel(channel=0).plot.contourf(levels=15, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False) #

    ax.set_title(f'{RP}-year storm')

plt.suptitle(f'[Independent] wind speeds for {calendar.month_name[month]} storms', y=1, fontsize='x-large')
plt.colorbar(im, ax=axs, extend='both', orientation='horizontal', label='Wind speed [m/s]',
             aspect=80, shrink=0.8)
# %% ----Visualise average winds for entire storm----
month = 10
median = data['median'].where(data['month']==month, drop=True).isel(channel=0)[0]
RP = (25,  50)

fig, axs = plt.subplots(1, 4, figsize=(18, 4), subplot_kw={'projection': ccrs.PlateCarree()})
RPs = [0, 5, 25, 50, 100]
for ax, i in zip(axs, range(len(RPs))):
    RP = (RPs[i], RPs[i+1])
    winds = data.where(data['storm_rp'] > RP[0], drop=True).where(data['storm_rp'] <= RP[1], drop=True).std(dim='sample', skipna=True)
    ax.coastlines(resolution='50m',color='k', linewidth=.5) 

    im = (median + winds.anomaly).isel(channel=0).plot(cmap=cmap, ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)
    (winds.anomaly + median).isel(channel=0).plot.contourf(levels=15, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False)

    ax.set_title(f'{RP[0]}-{RP[1]} year storms')

plt.suptitle(f'Std of wind speeds for {calendar.month_name[month]} storms', y=1, fontsize='x-large')
plt.colorbar(im, ax=axs, extend='both', orientation='horizontal', label='Wind speed [m/s]',
             aspect=80, shrink=0.8)
# %%
