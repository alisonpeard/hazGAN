"""
Plots mangrove damages and compares 1-in-10 year events.

1. Multiplies the damage probability with the mangrove area in each grid cell to get expected damage area.
2. Assign return periods based on expected mangrove damage.
3. Plot risk profile and damage probability fields.
"""
# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt

from utils.statistics import calculate_total_return_periods

bay_of_bengal_crs = 24346
FRACTION = False  # whether to divide by mangrove area

env = Env()
env.read_env()

# %% load generated data
samples_dir    = env.str("SAMPLES_DIR")
data_dir       = os.path.join("..", "..", "..", "hazGAN-data") #env.str("DATADIR")
mangroves_dir  = env.str("MANGROVE_DIR")
mangroves_path = env.str("MANGROVES")

train_damages       = xr.open_dataset(os.path.join(data_dir, "mangroves", "train_damages.nc"))
fake_damages        = xr.open_dataset(os.path.join(data_dir, "mangroves", "fake_damages.nc"))
independent_damages = xr.open_dataset(os.path.join(data_dir, "mangroves", "independent_damages.nc"))
dependent_damages   = xr.open_dataset(os.path.join(data_dir, "mangroves", "dependent_damages.nc"))

# different rates for different datasets
rate = train_damages.sizes["sample"] / 81
train_damages['rate']       = rate
fake_damages['rate']        = rate
independent_damages['rate'] = 1249 / 81

if True:
    # subset to 500 years of data
    nsamples = 500 * rate
    fake_mask = fake_damages["sample"] < nsamples
    fake_damages = fake_damages.isel(sample=fake_mask)

# %%
mangrove_grid_path = os.path.join(mangroves_dir, "mangrove_grid.nc")

if not os.path.exists(mangrove_grid_path):
    from shapely.geometry import box
    from utils import mangroveDamageModel

    #  Step 1: Clip mangroves to bay of bengal
    aoi = box(80.0, 10.0, 95.0, 25.0)  # Bay of Bengal bounding box
    mangroves = gpd.read_file(mangroves_path, mask=aoi)
    mangroves = mangroves.set_crs(epsg=4326).drop(columns='PXLVAL')
    mangroves['area']  = mangroves.to_crs(bay_of_bengal_crs).area

    # Step 2: Project mangroves to grid
    model = mangroveDamageModel()
    mangrove_grid = model.intersect_mangroves_with_grid(mangroves, train_damages.isel(sample=0))
    mangrove_grid.to_netcdf(mangrove_grid_path)

mangrove_grid = xr.open_dataset(mangrove_grid_path)
mangrove_area = mangrove_grid["area"].sum().values.item()
print(f"Total mangrove area: {mangrove_area:.2f} km²")

# %% Step 3:
train_damages['expected_damage']       = train_damages['damage_prob'] * mangrove_grid['area']
fake_damages['expected_damage']        = fake_damages['damage_prob'] * mangrove_grid['area']
independent_damages['expected_damage'] = independent_damages['damage_prob'] * mangrove_grid['area']
dependent_damages['expected_damage']   = dependent_damages['damage_prob'] * mangrove_grid['area']

# %%
tree = xr.DataTree()
tree['ERA5'] = xr.DataTree(train_damages)
tree['HazGAN']  = xr.DataTree(fake_damages)
tree['Independent'] = xr.DataTree(independent_damages)
tree.to_netcdf(os.path.join(mangroves_dir, "damage_areas.nc"))

# %%
def calculate_total_return_periods(
        damages:xr.Dataset, var:str,
        fraction:bool=FRACTION
        ) -> xr.Dataset:
    """
    Calculate total damages and return periods for a given variable
    in the damages dataset.
    """
    # skip root node in datatree
    if len(damages.data_vars) > 0:
        # aggregate to overall damages
        npy = damages['rate'].values.item()
        totals = damages[var].sum(dim=['lat', 'lon']).to_dataset()

        if fraction:
            totals = totals / mangrove_area
        
        # calculate return periods
        N = totals[var].sizes['sample']
        rank = totals[var].rank(dim='sample')
        totals['exceedence_prob'] = 1 - rank / (1 + N)
        totals['return_period'] = 1 / (npy * totals['exceedence_prob'])
        totals = totals.sortby('return_period')
        return totals

tree = tree.map_over_datasets(calculate_total_return_periods, 'expected_damage')
tree['Dependent'] = dependent_damages
tree['Dependent']['expected_damage'] = tree['Dependent']['expected_damage'].sum(dim=['lat', 'lon'])

if FRACTION:
    tree['Dependent']['expected_damage'] =  tree['Dependent']['expected_damage'] / mangrove_area


# %%
xmin = min([ds['return_period'].min() for ds in tree.values()]).data.item()
xmax = max([ds['return_period'].max() for ds in tree.values()]).data.item()
ymin = min([ds['expected_damage'].min() for ds in tree.values()]).data.item()
ymax = max([ds['expected_damage'].max() for ds in tree.values()]).data.item()

# %% PLOT THE LAMB (2010) style RISK PROFILE
from hazGAN.plotting.misc import blues

def riskprofileplot(tree:xr.DataTree, label:str, ax:plt.Axes,
                    minrp:float=1, maxrp:float=500, **kwargs):
    ds = tree[label].to_dataset()
    ds = ds.where(ds['return_period'] >= minrp, drop=True)
    ds = ds.where(ds['return_period'] <= maxrp, drop=True)
    print(ds["expected_damage"].shape)
    ax.plot(ds['return_period'], ds['expected_damage'], label=label, **kwargs)

plt.style.use('default')
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

scatter_kwargs = {'linestyle': 'none', 'marker': 'o', 'mfc': 'k',
                  'mec': 'k', 'mew': 0.25, 'alpha': 0.8, 'ms': 4}
line_kwargs = {'linewidth': 2, 'alpha': 0.8, 'marker': 'o', 'ms': 3} #, 'mfc': 'none'

fig, ax = plt.subplots(figsize=(7.0, 4.33))

riskprofileplot(tree, 'ERA5', ax, color='k', **scatter_kwargs)
riskprofileplot(tree, 'HazGAN', ax, color="#4682B4", **line_kwargs)
riskprofileplot(tree, 'Independent', ax, color=blues[3], zorder=0, **line_kwargs)
riskprofileplot(tree, 'Dependent', ax, color=blues[2], **line_kwargs)

def rp_damages(tree:xr.DataTree, label:str, rp:float) -> xr.Dataset:
    ds = tree[label].to_dataset()
    ds = ds.swap_dims({'sample': 'return_period'})
    damages = ds.sel(return_period=rp, method='nearest')["expected_damage"]
    return damages.values.item()

ax.vlines(100, ax.get_ylim()[0], rp_damages(tree, 'Dependent', 100), color='#333333', linestyle="--", alpha=0.4, linewidth=1, zorder=1)
ax.hlines(rp_damages(tree, 'ERA5', 100), 0, 100, color='#333333', linestyle="--", alpha=0.4, linewidth=1)
ax.hlines(rp_damages(tree, 'HazGAN', 100),0, 100, color='#4682B4', linestyle="--", alpha=0.4, linewidth=1)
ax.hlines(rp_damages(tree, 'Independent', 100),0, 100, color=blues[3], linestyle="--", alpha=0.4, linewidth=1)
ax.hlines(rp_damages(tree, 'Dependent', 100),0, 100, color=blues[2], linestyle="--", alpha=0.4, linewidth=1)

# legend options
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),  # Center horizontally, below the plot
    ncol=4,  # Spread entries horizontally
    frameon=False,
    handletextpad=0.5,  # Reduce space between handle and text (default is 0.8)
    columnspacing=2.0,  # Reduce space between columns (default is 2.0)
    labelspacing=0.4
)
plt.setp(ax.get_legend().get_title(), fontsize='12', fontweight='bold')

# configure x-axis
ax.set_xscale('log')
yticks = np.array([ymin, 2000, 4000, 6000, ymax])
yticks = yticks[(yticks > ymin) & (yticks <= ymax)]
yticklabels = [f"{y:.0f}" for y in yticks]

# ax.set_yticks(yticks, labels=yticklabels)
# ax.spines['left'].set_bounds(ymin, ymax)
# ax.text(0, ymin - 400, f"({ymin:.0f})", transform=ax.get_yaxis_transform(), 
#         ha='right', va='center', fontsize=12)

ax.set_xticks([1, 10, 100, 500], labels=['1', '10', '100', '500'])
ax.spines['bottom'].set_bounds(1, 500)

# ticks config
ax.tick_params(direction='in')
ax.xaxis.set_tick_params(which='minor', direction='in')
ax.yaxis.set_tick_params(which='minor', direction='in')
ax.tick_params(axis='x', length=8)
ax.tick_params(axis='y', length=8)
ax.tick_params(axis='x', which='minor', length=4)
ax.tick_params(axis='y', which='minor', length=4)

# turn off minor x-ticks
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.xaxis.set_minor_locator(plt.NullLocator())

ax.set_xlabel("Return period (years)", fontsize=13, fontweight='bold')
ax.set_ylabel("Expected\ndamage\narea\n(km$^2$)", 
              fontsize=12, 
              fontweight='bold',
              rotation=0,
              labelpad=15,  # Reduced padding
              va='center',
              ha='right')

# mark the 100-year return period
# ax.fill_betweenx([ymin, ymax], 0, 100, color="#F1F3F5", alpha=0.8, zorder=0) #'#F4F1EA'
# ax.axvline(x=100, ymax=0.95, color='#333333', linestyle='dashed', linewidth=1, zorder=1)


plt.subplots_adjust(left=0.2) 
plt.tight_layout()  
plt.savefig(os.path.join(data_dir, "..", "figures", 'risk_profile.pdf'), dpi=300, bbox_inches='tight')


# %% Damage probability field samples
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

# turn on spines
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True

RP = 20

def damagefield(
        tree:xr.DataTree, label:str,  ds, ds_var, axs:plt.Axes, rp=10,
        add_colorbar=False
        ) -> None:
    ds = ds.copy()
    ds["return_period"] = tree[label]['return_period']
    ds = ds.swap_dims({'sample': 'return_period'})
    ds = ds.sortby("return_period", ascending=True)
    sample = ds.sel(return_period=rp, method='nearest')
    
    # Plot on each row without colorbars
    im1 = sample.isel(field=0)[ds_var].plot.contourf(ax=axs[0], cmap="viridis", levels=10,
                                                     add_colorbar=False, vmin=0, vmax=32)
    im2 = sample.isel(field=1)[ds_var].plot.contourf(ax=axs[1], cmap="PuBu", levels=10, 
                                                     add_colorbar=False, vmin=0, vmax=0.3)
    im3 = sample['damage_prob'].plot.contourf(ax=axs[2], cmap="YlOrRd", levels=10,
                                                     add_colorbar=False, vmin=0, vmax=1)
    
    # Store the image references for later colorbar creation
    rp_approx = sample['return_period'].values.item()
    title = f"{label}\n(1-in-{rp_approx:.0f})"

    if add_colorbar:
        return im1, im2, im3, title
    
    axs[0].set_title(f"{label}\n(1-in-{rp_approx:.0f})")
    axs[1].set_title("")
    axs[2].set_title("")
    
    return title

# Create the figure with the appropriate layout
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.1, hspace=0.1)

# Create axes with projections
axs = []
for row in range(3):
    row_axes = []
    for col in range(4):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        row_axes.append(ax)
    axs.append(row_axes)
axs = plt.np.array(axs)

# Create separate axes for colorbars
cbar_ax1 = fig.add_subplot(gs[0, 4])
cbar_ax2 = fig.add_subplot(gs[1, 4])
cbar_ax3 = fig.add_subplot(gs[2, 4])

# Call damagefield for the first three columns without colorbars
title0 = damagefield(tree, 'ERA5', train_damages, 'train', axs[:, 0], rp=RP)
title1 = damagefield(tree, 'Dependent', dependent_damages, 'dependent', axs[:, 1], rp=RP)
title2 = damagefield(tree, 'Independent', independent_damages, 'independent', axs[:, 2], rp=RP)

# Call the last column with colorbar flag set to True to get the image references
im1, im2, im3, title3 = damagefield(tree, 'HazGAN', fake_damages, 'fake', axs[:, 3], add_colorbar=True, rp=RP)

# Add colorbars using the image references from the last column
plt.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='')
plt.colorbar(im2, cax=cbar_ax2, orientation='vertical', label='')
plt.colorbar(im3, cax=cbar_ax3, orientation='vertical', label='')

# format colorbars to have percentage labels
cbar_ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
cbar_ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
cbar_ax3.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, 0))

# Set labels and features for all axes
for ax in axs.flat:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2)

axs[0, 0].set_ylabel(r"Wind speed (ms$^{-1}$)")
axs[1, 0].set_ylabel("Precipitation (m)")
axs[2, 0].set_ylabel("Damage probability")

# Add titles to the top row
axs[-1, 0].set_title(title0, y=-.4)
axs[-1, 1].set_title(f"Dependent\n(1-in-{RP})", y=-.4)
axs[-1, 2].set_title(f"Independent\n(1-in-{RP})", y=-.4)
axs[-1, 3].set_title(f"HazGAN\n(1-in-{RP})", y=-.4)

plt.tight_layout()
plt.show()
fig.savefig(os.path.join(data_dir, "..", "figures", "10yr_mangrove_damages.pdf"), transparent=True, bbox_inches="tight")
# %% Error table

def assign_rps(
        tree:xr.DataTree, label:str, ds:xr.Dataset,
        rps:list=[5, 10, 25, 50, 100]
        ) -> xr.Dataset:
    ds = ds.copy()
    ds["return_period"] = tree[label]['return_period']
    ds["expected_damage"] = tree[label]['expected_damage']
    ds = ds.swap_dims({'sample': 'return_period'})
    ds = ds.sortby("return_period", ascending=True)
    sample = ds.sel(return_period=rps, method='nearest')
    return sample

all_rps = tree['ERA5']['return_period'].values.tolist()
sample_rps = [5, 10, 25, 50, 100, 500]
train_rps = assign_rps(tree, 'ERA5', train_damages, rps=sample_rps)
gan_rps = assign_rps(tree, 'HazGAN', fake_damages, rps=sample_rps)
indep_rps = assign_rps(tree, 'Independent', independent_damages, rps=sample_rps)
dep_rps = assign_rps(tree, 'Dependent', dependent_damages, rps=sample_rps)

# %% Make a table of expected damages and return periods
def make_risk_profile_table(
        rps:list, train_rps:xr.Dataset, gan_rps:xr.Dataset,
        indep_rps:xr.Dataset, dep_rps:xr.Dataset,
        fraction:bool=True
        ) -> pd.DataFrame:
    
    if fraction:
        constant = mangrove_area
    else:
        constant = 1.0
    data = []
    for rp in rps:

        # era5_rp = train_rps.sel(return_period=rp, method='nearest')
        # data.append(['ERA5', 'Expected damage area (km²)', rp, era5_rp['expected_damage'].values.item() / constant])
        # data.append(['ERA5', 'Return period (years)', rp, era5_rp['return_period'].values.item()])

        gan_rp = gan_rps.sel(return_period=rp, method='nearest')
        data.append(['HazGAN', 'Expected damage area (km²)', rp, gan_rp['expected_damage'].values.item() / constant])
        data.append(['HazGAN', 'Return period (years)', rp, gan_rp['return_period'].values.item()])

        indep_rp = indep_rps.sel(return_period=rp, method='nearest')
        data.append(['Independent', 'Expected damage area (km²)', rp, indep_rp['expected_damage'].values.item() / constant])
        data.append(['Independent', 'Return period (years)', rp, indep_rp['return_period'].values.item()])

        dep_rp = dep_rps.sel(return_period=rp, method='nearest')
        data.append(['Dependent', 'Expected damage area (km²)', rp, dep_rp['expected_damage'].values.item() / constant])
        data.append(['Dependent', 'Return period (years)', rp, dep_rp['return_period'].values.item()])
    return pd.DataFrame(data, columns=['Method', 'Deviation', 'Return Period', 'Value'])

df_risk_profile = make_risk_profile_table(
    sample_rps, train_rps, gan_rps, indep_rps, dep_rps, fraction=False
)

df_pivot = df_risk_profile.pivot_table(
    index=['Method', 'Deviation'],
    columns='Return Period',
    values='Value'
)
# Print the DataFrame
print(df_pivot.to_string(float_format='%.2f'))


# %% Calculate errors
gan_err = gan_rps["expected_damage"].values - train_rps["expected_damage"].values
gan_dev = gan_rps["return_period"].values - train_rps["return_period"].values
indep_err = indep_rps["expected_damage"].values - train_rps["expected_damage"].values
indep_dev = indep_rps["return_period"].values - train_rps["return_period"].values
dep_err = dep_rps["expected_damage"].values - train_rps["expected_damage"].values
dep_dev = dep_rps["return_period"].values - train_rps["return_period"].values

gan_err   = list(gan_err)
indep_err = list(indep_err)
dep_err   = list(dep_err)
gan_dev   = list(gan_dev)
indep_dev = list(indep_dev)
dep_dev   = list(dep_dev)

# %% get mean absolute error 
train_rps = assign_rps(tree, 'ERA5', train_damages, rps=all_rps)
gan_rps = assign_rps(tree, 'HazGAN', fake_damages, rps=all_rps)
indep_rps = assign_rps(tree, 'Independent', independent_damages, rps=all_rps)
dep_rps = assign_rps(tree, 'Dependent', dependent_damages, rps=all_rps)

def calculate_mae(ds1, ds2, var):
    """Calculate mean absolute error between two datasets."""
    return np.mean(np.abs(ds1[var].values - ds2[var].values))

mae_gan = calculate_mae(train_rps, gan_rps, 'expected_damage')
mae_indep = calculate_mae(train_rps, indep_rps, 'expected_damage')
mae_dep = calculate_mae(train_rps, dep_rps, 'expected_damage')

mae_gan_rp = calculate_mae(train_rps, gan_rps, 'return_period')
mae_indep_rp = calculate_mae(train_rps, indep_rps, 'return_period')
mae_dep_rp = calculate_mae(train_rps, dep_rps, 'return_period') 

# append the MAE errors
gan_err += [mae_gan]
indep_err += [mae_indep]
dep_err += [mae_dep]
gan_dev += [mae_gan_rp]
indep_dev += [mae_indep_rp]
dep_dev += [mae_dep_rp]
sample_rps.append('MAE')  # Add MAE to the list of return periods

# %% Create hierarchical data structure
data = []

for i, rp in enumerate(sample_rps):
    data.append(['HazGAN', 'Expected damage area (km²)', rp, gan_err[i]])
    data.append(['HazGAN', 'Return period (years)', rp, gan_dev[i]])

    data.append(['Independent', 'Expected damage area (km²)', rp, indep_err[i]])
    data.append(['Independent', 'Return period (years)', rp, indep_dev[i]])

    data.append(['Dependent', 'Expected damage area (km²)', rp, dep_err[i]])
    data.append(['Dependent', 'Return period (years)', rp, dep_dev[i]])


df_long = pd.DataFrame(data, columns=['Method', 'Deviation', 'Return Period', 'Value'])
df_pivot = df_long.pivot_table(index=['Method', 'Deviation'], 
                               columns='Return Period', 
                               values='Value')

formatter = lambda x: f'{x} yr' if isinstance(x, int) else x
df_pivot.columns = [formatter(col) for col in df_pivot.columns]


# change row order
df_pivot = df_pivot.reindex(
    index=[
        ('HazGAN', 'Expected damage area (km²)'),
        ('HazGAN', 'Return period (years)'),
        ('Independent', 'Expected damage area (km²)'),
        ('Independent', 'Return period (years)'),
        ('Dependent', 'Expected damage area (km²)'),
        ('Dependent', 'Return period (years)')
    ]
)
print(df_pivot.to_string(float_format='%.2f'))

# %%
latex_table = df_pivot.to_latex(
    float_format='%.2f',
    escape=False,
    header=True,
    multirow=True,
    multicolumn_format='c',
    caption="Expected damage area and return period deviations for GAN, independent, and dependent generated samples.",
    label="tab:riskprofile"
)

print(latex_table)
# %%
