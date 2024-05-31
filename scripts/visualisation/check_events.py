
# %%
"""Using env hazGAN """
import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from hazGAN import xmin, xmax, ymin, ymax

def extract_timerange(df, event):
    df = df[df['storm'] == event].copy()
    return df['time'].min(), df['time'].max()

datadir = os.path.expandvars("$HOME/Documents/DPhil/multivariate/era5_data")
#%% Load event data
events_df = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
events_df['time'] = pd.to_datetime(events_df['time'])
events_df = events_df[events_df['time'].dt.year == 2012]
event_indices = [*events_df['storm'].unique()]
event_data = {event: extract_timerange(events_df, event) for event in event_indices}
# %% Look at event duration distribution
events_df = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
events_df['time'] = pd.to_datetime(events_df['time'])
start, stop = events_df['time'].min(), events_df['time'].max()
fig, ax = plt.subplots(1,1, figsize=(6.5, 4))
events_df['storm.size'].hist(color='lightblue', edgecolor='k', bins=50, density=True, ax=ax)
ax.set_xlabel('Days')
ax.set_ylabel('Density')
fig.suptitle('Distribution of event durations')
ax.set_title(f"{start.year}-{stop.year}")
# %% Extremeness vs. duration
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
ax.scatter(events_df['storm.rp'], events_df['storm.size'], color='k', s=1)
ax.set_xlabel('Empirical return period')
ax.set_ylabel('Duration (days)')
fig.suptitle('Event duration vs. return period')
ax.set_title(f"{start.year}-{stop.year}")
# %%
# events_df['ecdf'].hist(kind='')
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

sns.histplot(data=events_df['storm.rp'], ax=ax)
ax.set_yscale('log')
ax.set_title('Empirical return period distribution')
# %% Extremeness vs. duration (jointplot)
if False: # these take ages
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    sns.jointplot(x='ecdf', y='storm.size', data=events_df, kind='kde', fill=True, cmap='Blues', ax=ax) #
    ax.set_xlabel('Extremeness')
    ax.set_ylabel('Duration (days)')
    fig.suptitle('Event duration vs. extremeness')
    ax.set_title(f"{start.year}-{stop.year}")
# %% Distribution of lags
extremes_df = pd.read_parquet(os.path.join(datadir, "fitted_data.parquet"))
extremes_df['time.u10'] = pd.to_datetime(extremes_df['time.u10'])
extremes_df['time.mslp'] = pd.to_datetime(extremes_df['time.mslp'])
extremes_df['lag'] = pd.to_numeric(extremes_df['time.u10'] - extremes_df['time.mslp'])
# %% plot distribution of lags
fig, ax = plt.subplots(1,1, figsize=(6.5, 4))
extremes_df['lag'].hist(color='lightblue', edgecolor='k', bins=100, density=False, ax=ax)
ax.set_xlabel('Days')
ax.set_ylabel("Count")
# ax.set_yscale('log')
fig.suptitle(r'Lags $(T^\max_{U10} - T^\min_{MSLP})$')
ax.set_title(f"{start.year}-{stop.year}")
# %% Joint-distribution of maxima
fig, ax = plt.subplots(1,1)
ax.scatter(extremes_df['u10'], extremes_df['msl'], color='k', s=1)
ax.set_xlabel('10m wind')
ax.set_ylabel('Mean sea level pressure')
ax.set_title('storm maxima')
fig.suptitle('1950-2022)')

#%% Compare to IBTrACS hurricane records
IBTRACS_AGENCY_10MIN_WIND_FACTOR = {"wmo": [1.0, 0.0],
                                    "usa": [1.0, 0.0], "tokyo": [1.0, 0.0],
                                    "newdelhi": [0.88, 0.0], "reunion": [1.0, 0.0],
                                    "bom": [1.0, 0.0], "nadi": [1.0, 0.0],
                                    "wellington": [1.0, 0.0], 'cma': [1.0, 0.0],
                                    'hko': [1.0, 0.0], 'ds824': [1.0, 0.0],
                                    'td6': [1.0, 0.0], 'td5': [1.0, 0.0],
                                    'neumann': [1.0, 0.0], 'mlc': [1.0, 0.0],
                                    'hurdat_atl' : [0.88, 0.0], 'hurdat_epa' : [0.88, 0.0],
                                    'atcf' : [0.88, 0.0],     'cphc': [0.88, 0.0]
}

ibtracs = gpd.read_file(os.path.expandvars("$HOME/Documents/DPhil/data/ibtracs/ibtracs_since1980_points.gpkg"),
                        bbox=[xmin, ymin, xmax, ymax])
ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
wind_cols = [col for col in ibtracs.columns if 'WIND' in col.upper()]
factors = {key[:3]: value for key, value in IBTRACS_AGENCY_10MIN_WIND_FACTOR.items()}
for col in wind_cols:
    agency = col.split('_')[0].lower()
    scale, shift = factors[agency]
    ibtracs[col] = ibtracs[col] * scale + shift
ibtracs['wind'] = ibtracs[wind_cols].max(axis=1) * 0.514 # knots to mps
ibtracs = ibtracs[['time', 'NAME', 'wind', 'LAT', 'LON']]
ibtracs.columns = ['time', 'event', 'wind', 'lat', 'lon']
ibtracs = ibtracs.groupby(pd.Grouper(key='time', freq='D')).agg({'event': 'first', 'wind': 'max', 'lat': 'mean', 'lon': 'mean'}).reset_index()
ibtracs['event'] = ibtracs['event'].fillna('NOT_NAMED').apply(lambda s: s.title())
ibtracs = ibtracs.dropna(subset=['wind'])
amphan = ibtracs[ibtracs['event'] == 'Amphan']
# %%
amphan = gpd.GeoDataFrame(amphan, geometry=gpd.points_from_xy(amphan.lon, amphan.lat)).drop(columns=['lat', 'lon'])
amphan = amphan.set_crs(epsg=4326)
# %%
event_df = pd.read_parquet(os.path.join(datadir, "event_data.parquet"))
events_df['time'] = pd.to_datetime(events_df['time'])
events_df = events_df[events_df['time'].dt.year >= 1980].copy()
events_df = events_df[['time', 'storm']]
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
ax.set_xlabel("Count [days]")
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
