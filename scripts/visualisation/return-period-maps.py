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

datadir = '/Users/alison/Documents/DPhil/multivariate/era5_data'
data = xr.open_dataset(os.path.join(datadir, "hazGAN_samples.nc"))
occurence_rate = 22 # from R 
# %% calculate return periods across each pixel
wind = data.anomaly.sel(channel='u10')
m = wind.sample.size
exceed_prob = 1 - (wind.rank(dim='sample') / (m + 1))
rp = 1 / (occurence_rate * exceed_prob)
wind['return_period'] = rp
wind['aep'] = 1 / rp
# %% interpolate return periods
def get_rp_wind(x, y, rp):
    interpolated = np.interp(rp, x, y)
    return interpolated

res = xr.apply_ufunc(get_rp_wind, rp, wind,
                     input_core_dims=[["sample"], ['sample']],
                     dask = 'allowed',
                     kwargs={'rp': 100},
                     vectorize = True)
# %% visualise different months
month = 8
median = data['median'].where(data['month']==month, drop=True).isel(channel=0)[0]#.mean(dim=['time']).medians

fig = plt.figure(figsize=(8, 6))
ax  = plt.axes(projection=ccrs.PlateCarree()) 
ax.coastlines(resolution='50m',color='k', linewidth=.5) 
(median + res).plot.contourf(levels=20, cmap="YlOrRd", ax=ax)

ax.set_title(f'50 year RP wind speed for {calendar.month_name[month]} storm')
# %%

