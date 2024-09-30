# %%
import os
import xarray as xr
import geopandas as gpd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']
plt.style.use('bmh')

# %%
wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')

totals_train = xr.open_dataset(os.path.join(wd, 'results', 'mangroves', 'totals.nc'), group='train')
totals_test = xr.open_dataset(os.path.join(wd, 'results', 'mangroves', 'totals.nc'), group='test')
totals_dependent = xr.open_dataset(os.path.join(wd, 'results', 'mangroves', 'totals.nc'), group='dependent')
totals_independent = xr.open_dataset(os.path.join(wd, 'results', 'mangroves', 'totals.nc'), group='independent')
totals_hazGAN = xr.open_dataset(os.path.join(wd, 'results', 'mangroves', 'totals.nc'), group='hazGAN')

totals_train = totals_train.rename({'era5_damagearea': 'damagearea'})
totals_test = totals_test.rename({'era5_damagearea': 'damagearea'})
totals_dependent = totals_dependent.rename({'dependent_damagearea': 'damagearea'})
totals_independent = totals_independent.rename({'independent_damagearea': 'damagearea'})
totals_hazGAN = totals_hazGAN.rename({'hazGAN_damagearea': 'damagearea'})

# %%
eps = .25

def plot_totals(totals, ax, label, color, eps=.25, scatter=True,
                line=True, line_kws={}, scatter_kws={}):
    mask = totals.return_period > eps
    if line:
        ax.plot(
            totals.where(mask).return_period,
            totals.where(mask).damagearea,
            color=color,
            linewidth=1.5,
            label=label,
            **line_kws
        )
    if scatter:
        ax.scatter(
            totals.where(mask).return_period,
            totals.where(mask).damagearea,
            color=color,
            s=1.5,
            label=label,
            **scatter_kws
        )

fig, ax = plt.subplots()
plot_totals(totals_test, ax, 'Test data', 'c', line=False, scatter_kws={'marker': '+'})
plot_totals(totals_train, ax, 'Training data', 'k', line=False)

plot_totals(totals_dependent, ax, 'Complete dependence', 'b', scatter=False, line_kws={'linestyle': 'dotted'})
plot_totals(totals_independent, ax, 'Independence', 'r', scatter=False, line_kws={'linestyle': 'dashed'})
plot_totals(totals_hazGAN, ax, 'Modelled dependence', 'k', scatter=False)

ax.set_xlabel('Return period (years)')
ax.set_ylabel('Total damage to mangroves (km²)')
ax.legend()

ax.set_xscale('log')
ax.set_xticks([1, 2, 5, 25, 100, 200, 500])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter());
# %%

# %%
