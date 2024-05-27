# %%import os
import xarray as xr

# %%
ds = xr.open_dataset('/Users/alison/Documents/DPhil/multivariate/era5_data/data.nc')
ds.isel(time=0, channel=0).uniform.plot()
ds['u10'] = ds.isel(channel=0).medians + ds.isel(channel=0).anomaly
# %%
wind_maxima = ds['u10'].max(dim=['lon', 'lat'])
yearly_maxima = wind_maxima.groupby('time.year').max()
# %%
import matplotlib.pyplot as plt
plt.hist(yearly_maxima, bins=50, color='lightgrey', edgecolor='k')
# %% ECDF
n = len(yearly_maxima)
yearly_maxima = yearly_maxima.sortby(yearly_maxima)
print("Number of years: ", n)
ecdf = yearly_maxima.rank(dim='year') / (n + 1)
empirical_rp = 1 / (1-ecdf)
# %% GEV
from scipy.stats import genextreme as gev
params = gev.fit(yearly_maxima)
cdf = gev.cdf(yearly_maxima, *params)
parametric_rp = 1 / (1-cdf)
# %% plot results
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(yearly_maxima, ecdf, label='Empirical')
axs[0].plot(yearly_maxima, cdf, label='GEV')
axs[0].set_title("CDF")
axs[0].legend()
axs[0].set_xlabel("10m wind maximum speed (m/s)")
axs[0].set_ylabel("F(x)=P(X<=x)")

axs[1].plot(empirical_rp, yearly_maxima, label='Empirical')
axs[1].plot(parametric_rp, yearly_maxima, label='GEV')
axs[1].set_title("Return Period")
axs[1].legend()
axs[1].set_ylabel("10m wind maximum speed (m/s)")
axs[1].set_xlabel("Return Period (years)")
plt.suptitle("Bangladesh 10m wind speed return periods (yearly maxima)")
# %% now apply to wind_maxima
import numpy as np
wind_maxima = wind_maxima.sortby(wind_maxima)
cdf = gev.cdf(wind_maxima, *params)
ecdf = np.interp(wind_maxima, yearly_maxima, ecdf)
parametric_rp = 1 / (1-cdf)
empirical_rp = np.interp(wind_maxima, yearly_maxima, empirical_rp)

# %%
ecdf_all = wind_maxima.rank(dim='time') / (len(wind_maxima) + 1)
empirical_rp_all = 1 / (1-ecdf_all)
params_all = gev.fit(wind_maxima)
cdf_all = gev.cdf(wind_maxima, *params_all)
parametric_rp_all = 1 / (1-cdf_all)
fig, axs = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

color = 'tab:blue'
axs[0].plot(wind_maxima, 1- cdf, label='GEV', color=color)
axs[0].scatter(wind_maxima, 1 - ecdf, label='Empirical', s=2, color=color)
axs[0].set_ylabel("P(X>x) (per year)", color=color)
axs[0].tick_params(axis='y', labelcolor=color)
axs[0].set_xlabel("10m wind maximum speed (m/s)")
axs[0].legend()

axs1 = axs[0].twinx()
color='tab:red'
axs1.plot(wind_maxima, 1 - cdf_all, label='GEV (all data)', color=color)
axs1.scatter(wind_maxima, 1 - ecdf_all, label='Empirical (all data)', s=2, color=color)
axs1.set_ylabel("P(X>x) (per event)", color=color)
axs1.tick_params(axis='y', labelcolor=color)

color='tab:blue'
axs[1].plot(wind_maxima, parametric_rp, label='GEV', color=color)
axs[1].scatter(wind_maxima, empirical_rp, label='Empirical', s=2, color=color)
axs[1].set_ylabel("Return Period (#years)", color=color)
axs[1].tick_params(axis='y', labelcolor=color)
axs[1].set_xlabel("10m wind maximum speed (m/s)")
axs[1].axvline(25, color='k', linestyle='--', label='25 mps')
axs[1].axhline(6, color=color, linestyle='--')
axs[1].legend()

# now compare to event-wise
axs2 = axs[1].twinx()
color='tab:red'
axs2.scatter(wind_maxima, empirical_rp_all, label='Empirical (all data)', s=2, color=color)
axs2.plot(wind_maxima, parametric_rp_all, label='GEV (all data)', color=color)
axs2.tick_params(axis='y', labelcolor=color)
axs2.axhline(50, color=color, linestyle='--')
axs2.set_ylabel("Return Period (#events)", color=color)

plt.suptitle("Bay of Bengal 10m wind speed return periods")
# %% now apply to wind_maxima