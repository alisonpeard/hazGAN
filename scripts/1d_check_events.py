
# %%
"""Using env tf_geo """
import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_timerange(df, event):
    df = df[df['cluster'] == event].copy()
    return df['time'].min(), df['time'].max()

datadir = os.path.expandvars("$HOME/Documents/DPhil/multivariate/era5_data")
#%% Load event data
events_df = pd.read_csv(os.path.join(datadir, "event_data.csv"))
events_df['time'] = pd.to_datetime(events_df['time'])
events_df = events_df[events_df['time'].dt.year == 2012]
event_indices = [*events_df['cluster'].unique()]
event_data = {event: extract_timerange(events_df, event) for event in event_indices}
# %% Look at event duration distribution
events_df = pd.read_csv(os.path.join(datadir, "event_data.csv"))
events_df['time'] = pd.to_datetime(events_df['time'])
start, stop = events_df['time'].min(), events_df['time'].max()
fig, ax = plt.subplots(1,1, figsize=(6.5, 4))
events_df['cluster.size'].hist(color='lightblue', edgecolor='k', bins=50, density=True, ax=ax)
ax.set_xlabel('Days')
ax.set_ylabel('Density')
fig.suptitle('Distribution of event durations')
ax.set_title(f"{start.year}-{stop.year}")
# %% Extremeness vs. duration
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
ax.scatter(events_df['ecdf'], events_df['cluster.size'], color='k', s=1)
ax.set_xlabel('Extremeness')
ax.set_ylabel('Duration (days)')
fig.suptitle('Event duration vs. extremeness')
ax.set_title(f"{start.year}-{stop.year}")
# %%
# events_df['ecdf'].hist(kind='')
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

sns.histplot(data=events_df['ecdf'], kde=True, ax=ax)
ax.set_title('Extremeness distribution')
# %% Extremeness vs. duration (jointplot)
if False: # these take ages
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    sns.jointplot(x='ecdf', y='cluster.size', data=events_df, kind='kde', fill=True, cmap='Blues', ax=ax) #
    ax.set_xlabel('Extremeness')
    ax.set_ylabel('Duration (days)')
    fig.suptitle('Event duration vs. extremeness')
    ax.set_title(f"{start.year}-{stop.year}")
# %% Distribution of lags
extremes_df = pd.read_csv(os.path.join(datadir, "fitted_data.csv"))
extremes_df['time.u10'] = pd.to_datetime(extremes_df['time.u10'])
extremes_df['time.mslp'] = pd.to_datetime(extremes_df['time.mslp'])
extremes_df['lag'] = pd.to_numeric(extremes_df['time.u10'] - extremes_df['time.mslp'])
# %% plot distribution of lags
fig, ax = plt.subplots(1,1, figsize=(6.5, 4))
extremes_df['lag'].hist(color='lightblue', edgecolor='k', bins=100, density=False, ax=ax)
ax.set_xlabel('Days')
ax.set_ylabel("Count")
# ax.set_yscale('log')
fig.suptitle(r'Lags $(T^\max_{U10} - T^\max_{MSLP})$')
ax.set_title(f"{start.year}-{stop.year}")
# %% Joint-distribution of maxima
fig, ax = plt.subplots(1,1)
ax.scatter(extremes_df['u10'], extremes_df['msl'], color='k', s=1)
ax.set_xlabel('10m wind')
ax.set_ylabel('Mean sea level presseure')
ax.set_title('Cluster maxima')
fig.suptitle('1950-2022)')
# %% Joint-distribution of maxima (jointplot)§
if False: # these take ages
    sns.jointplot(x='u10', y='msl', data=extremes_df, kind='kde', fill=True, cmap='Blues')
    sns.jointplot(x='u10', y='msl', data=extremes_df, kind='hex', cmap='Blues')
# %% Compare to historical events
# Cyclone Nilam October 28, 2012 - October 31, 2012
historical = {'nilim': ('2012-10-28', '2012-10-31')}
#%% Compare to IBTrACS hurricane records
from hazardGANv2 import xmin, xmax, ymin, ymax

ibtracs = gpd.read_file(os.path.expandvars("$HOME/Documents/DPhil/data/ibtracs/ibtracs_since1980_points.gpkg"))
ibtracs = ibtracs.clip([xmin, ymin, xmax, ymax])
ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
ibtracs = ibtracs[['time', 'NAME']]
ibtracs.columns = ['time', 'event']
ibtracs = ibtracs.groupby(pd.Grouper(key='time', freq='D')).first().reset_index()
#%%
events_df = pd.read_csv(os.path.join(datadir, "event_data.csv"))
events_df['time'] = pd.to_datetime(events_df['time'])
events_df = events_df[events_df['time'].dt.year >= 1980].copy()
events_df = events_df[['time', 'cluster']]
events_df.columns = ['time', 'event']
assert events_df['time'].dt.year.max() == ibtracs['time'].dt.year.max(), "Must have same time range."

merged = ibtracs.merge(events_df, on='time', how='outer', suffixes=('_ibtracs', '_era5'))
merged = merged.sort_values(by='time').set_index('time')
merged['event_ibtracs'] = merged['event_ibtracs'].notnull()
merged['event_era5'] = merged['event_era5'].notnull()
# %% confusion matrix
def confusion_score(row):
    if row['event_ibtracs'] and row['event_era5']:
        return 'TP'
    elif row['event_ibtracs'] and not row['event_era5']:
        return 'FN'
    elif row['event_era5'] and not row['event_ibtracs']:
        return 'FP'
    elif not row['event_ibtracs'] and not row['event_era5']:
        return 'TN'
    else:
        raise ValueError("Invalid row")
    
merged['confusion_score'] = merged.apply(confusion_score, axis=1)
# %% plot confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
merged.value_counts('confusion_score').plot(kind='barh', color='lightblue', edgecolor='k', ax=ax)
ax.set_ylabel('Confusion score')
ax.set_title('Are we picking up IBTrACS hurricanes?')
ax.set_xlabel("Count")
# %% print confusion metrics
precision = merged['confusion_score'].value_counts()['TP'] / (merged['confusion_score'].value_counts()['TP'] + merged['confusion_score'].value_counts()['FP'])
recall = merged['confusion_score'].value_counts()['TP'] / (merged['confusion_score'].value_counts()['TP'] + merged['confusion_score'].value_counts()['FN'])
f1 = 2 * (precision * recall) / (precision + recall)

print(events_df['event'].max(), "events")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
# %% Save IBTrACs dates and compare in R
ibtracs = ibtracs.sort_values(by='time')
ibtracs.to_csv(os.path.join(datadir, 'ibtracs_dates.csv'), index=False)
# %%
