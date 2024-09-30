#%%
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
#%%
# Inv PIT plot
x = stats.uniform(0, 1).rvs(10000)
norm = stats.distributions.norm()
x_trans = norm.ppf(x)
h = sns.jointplot(x=x, y=x_trans)
h.set_axis_labels(r'$u$', r'$F^{-1}(u)$', fontsize=16);
# %%
# PIT plot
# make background transparent
plt.rcParams['axes.facecolor'] = 'none'
h = sns.jointplot(x=x_trans, y=x)
h.set_axis_labels(r'$x$', r'$F(x)$', fontsize=16);

# save with transparent background
h.savefig('PIT_plot.png', transparent=True)

# %% PIT Gumbel
red = '#F52224'
m1 = stats.gumbel_l()
x_gumbel = -m1.ppf(x)
h = sns.jointplot(x=x_gumbel, y=x,
    color='k',
    marker='.',
    edgecolor='k',
    marginal_kws={
        'color': red,
        'edgecolor': 'white'
        });    
h.set_axis_labels(r'$x$', r'$F(x)$', fontsize=16);
plt.savefig(os.path.join('/Users/alison/Documents/DPhil/paper1.nosync/figures/paper', 'Gumbel_PIT.png'), transparent=True)
# %%
mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5], 
                                                     [0.5, 1.]])
# Generate random samples from multivariate normal with correlation .5
fig, ax = plt.subplots(figsize=(5, 5))
x = mvnorm.rvs(100000)
h = sns.jointplot(x=x[:,0], y=x[:,1], kind='kde', fill=True, cmap='Blues', ax=ax);
h.set_axis_labels('variable 1', 'variable 2', fontsize=18);
h.figure.suptitle('Bivariate Gaussian', fontsize=20, y=1.05)
# %%
# Uniform-uniform
norm = stats.norm()
x_unif = norm.cdf(x)
fig, ax = plt.subplots(figsize=(5, 5))
h = sns.jointplot(x=x_unif[:, 0], y=x_unif[:, 1], kind='hex', ax=ax) # , fill=True, cmap='Blues'
h.set_axis_labels('variable 1', 'variable 2', fontsize=18);
h.figure.suptitle('Transformed to uniform', fontsize=20, y=1.05)
# %%
# Two Gumbel plots
m1 = stats.gumbel_l()
m2 = stats.beta(a=10, b=2)

x1_trans = m1.ppf(x_unif[:, 0])
x2_trans = m2.ppf(x_unif[:, 1])
# %%
# All blue
fig, ax = plt.subplots(figsize=(5, 5))
h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), fill=True, cmap='Blues', ax=ax);
h.set_axis_labels('variable 1', 'variable 2', fontsize=18);
h.figure.suptitle('Transformed to Gumbel', fontsize=20, y=1.05)
# %%
# Red marginals
fig, ax = plt.subplots(figsize=(5, 5))
h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), fill=True, cmap='Blues',
                  marginal_kws={'color': 'red'}, ax=ax);
h.set_axis_labels('variable 1', 'variable 2', fontsize=16);
h.figure.suptitle('Transformed to Gumbel', fontsize=16, y=1.05)
# %%
# Red dependence
fig, ax = plt.subplots(figsize=(5, 5))
h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), fill=True, cmap='Reds',
                  ax=ax);
h.set_axis_labels('variable 1', 'variable 2', fontsize=16);
h.figure.suptitle('Transformed to Gumbel', fontsize=16, y=1.05)
#%%
# Independent Gumbels
x1 = m1.rvs(10000)
x2 = m2.rvs(10000)

fig, ax = plt.subplots(figsize=(5, 5))
h = sns.jointplot(x=x1, y=x2, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), fill=True, cmap='Blues', ax=ax);
h.set_axis_labels('variable 1', 'variable 2',  fontsize=18);
h.figure.suptitle('Independent Gumbels', fontsize=20, y=1.05)
# %%
###############################################################################
# HURRICANE PLOT
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

data = pd.read_csv("/Users/alison/Documents/DPhil/multivariate/wind_data/daily/total_dailymax.csv", index_col=0)
coords = gpd.read_file('/Users/alison/Documents/DPhil/multivariate/wind_data/coords.gpkg', index_col='grid')
coords['grid'] = coords['grid'].astype(int)
coords = coords.set_index('grid')   
data = pd.melt(data, id_vars=['time', 'cyclone_flag'], value_vars=data.columns[2:], value_name='u10', var_name='grid')
data['grid'] = data['grid'].astype(int)
# data = data[data['cyclone_flag'] == 1]
gdf = gpd.GeoDataFrame(data.join(coords, on='grid', how='left'), geometry='geometry')
gdf['time'] = pd.to_datetime(gdf['time'])
# %%#
# use IBTrACs data
xmin = 80.0
xmax = 95.0
ymin = 10.0
ymax = 25.0
mask = box(xmin, ymin, xmax, ymax)
ibtracs = gpd.read_file('/Users/alison/Documents/DPhil/data/ibtracs/ibtracs_since1980_points.gpkg', mask=mask)
ibtracs = ibtracs[ibtracs['NAME'] != 'NOT_NAMED']
ibtracs = ibtracs.sort_values(['ISO_TIME', 'NAME'])
ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
ibtracs = ibtracs[['time', 'NAME']].groupby('NAME').agg(['first', 'last'])
ibtracs = ibtracs.reset_index()
ibtracs.columns = ['Name', 'start', 'end']
ibtracs = ibtracs[ibtracs['start'].dt.year > 2013].reset_index(drop=True)
# %%
import matplotlib.pyplot as plt
i = 0
name = ibtracs.loc[i, 'Name']
start = ibtracs.loc[i, 'start']
end = ibtracs.loc[i, 'end']
event = gdf[(gdf['time'] >= start) & (gdf['time'] <= end)]
event = event.sort_values('u10', ascending=True) # so highest winds are plotted on top
####
event_max = gpd.GeoDataFrame(event[['time', 'geometry', 'u10']].groupby(['time', 'geometry']).max().reset_index(), geometry='geometry')
event_max = event_max.sort_values('u10', ascending=True)
vmin = event['u10'].min()
vmax = event['u10'].max()

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
event_max.plot('u10', vmin=vmin, vmax=vmax, ax=axs, legend=True)
axs.set_title("Cyclone {}".format(name.lower().capitalize()));
#####
event['day'] = event['time'].dt.dayofyear
event = event[['day', 'geometry', 'u10']].groupby(['day', 'geometry']).max().reset_index()
ndays = event['day'].nunique()
fig, axs = plt.subplots(1, ndays, figsize=(5*ndays, 3))
for j, day in enumerate(event['day'].unique()):
    event_day = gpd.GeoDataFrame(event[event['day'] == day], geometry='geometry')
    event_day.plot('u10', vmin=vmin, vmax=vmax, ax=axs[j], legend=True)
    axs[j].set_title("Day {}".format(j));

# fig, axs = p
# %%

# %%


ibtracs.loc[0, 'Name']
# Viz


# %%
# join by grid in coords and index in data
gdf['time'] = pd.to_datetime(gdf['time'])
gdf = gdf[gdf['time'] == gdf['time'].min()]
# %%
gdf.plot('u10', legend=True)
# %%
