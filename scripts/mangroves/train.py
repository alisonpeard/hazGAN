"""
To load model elsewhere, you can use the following snippet:
>>> from joblib import load
>>> modelpath = '/Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/model.pkl'
>>> with open(modelpath, "rb") as f:
>>>     model = load(f)
"""
#%%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import dump # for pickling
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

global scaling
scaling = False
visuals = False
# %%
def transform(df: pd.DataFrame, X: list, y: str):
 # shift transform and scale
 if scaling:
  df = df[[y] + X].dropna()
  df[X] = df[X].apply(shift_positive, axis=0)
  df = np.log10(df).dropna()
 return(df)


def inverse_transform(y: pd.Series):
  if scaling:
    y = 10**y
  return(y)


def shift_positive(x: pd.Series):
  if x.min() <= 0:
    x += abs(x.min()) + 1
  return x


def rsquared(y, yhat):
    """Psuedo R^2"""
    ymean = np.mean(y)
    y = np.where(y == ymean, np.nan, y)
    return sum((yhat - ymean)**2) / sum((y - ymean)**2)


def rsquared_res(y, yhat):
  rss = sum((y - yhat)**2)
  tss = sum((y - np.mean(y))**2)
  if tss > 0:
    return 1 - rss / tss
  else:
    return np.nan


def mean_average_percent_error(y, yhat):
    return np.mean(np.abs((y - yhat) / y))


def mean_squared_error(y, yhat):
    return np.mean((y - yhat)**2)


bob_crs = 24346
indir_alison = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'
indir_yu = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2'
infile1 = os.path.join(indir_alison, 'data_with_slopes.csv')
infile2 = os.path.join(indir_alison, 'final.csv')
infile3 = os.path.join(indir_alison, 'era5_and_slope_0.csv')
infile0 = os.path.join(indir_yu, 'result/model/input_fixedCNTRY_rmOutlier.csv')

infile = infile2
response = 'intensity'
regressor_rename = {
    'totalPrec_total': 'precip',
    # 'slope': 'slope',
    'landingWindMaxLocal2': 'wind'
    }
regressors = [value for value in regressor_rename.values()]
scaler = StandardScaler()

# %% load and process data
df = pd.read_csv(infile).rename(columns=regressor_rename)
events = [*df['landing'].unique()]

df_transformed = transform(df, regressors, response)
df_transformed[['landing', 'center_centerLon', 'center_centerLat']] = df[['landing', 'center_centerLon', 'center_centerLat']]
gdf = gpd.GeoDataFrame(df_transformed, geometry=gpd.points_from_xy(df_transformed.center_centerLon, df_transformed.center_centerLat)).set_crs(epsg=4326)

df = df.iloc[gdf.index] # align indices
# %% ----Train model eventwise OOB---- 
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def ensemble(X_train, y_train, X_test):
  xgb = XGBRegressor(n_estimators=15).fit(X_train, y_train)
  xgb_fit = xgb.predict(X_train).reshape(-1, 1)
  xgb_pred = xgb.predict(X_test).reshape(-1, 1)
  linear = LinearRegression().fit(xgb_fit, y_train)
  y_fit = linear.predict(xgb_fit)
  y_pred = linear.predict(xgb_pred)
  return y_pred


results = df[['landing', 'intensity']].rename(columns={'intensity': 'y'})
predictions = []
mses = []
rsqs = []

# loop through events
for event in tqdm(events):
  gdf_train = gdf[gdf['landing'] != event]
  gdf_test = gdf[gdf['landing'] == event]
  y_fullscale = df[df['landing'] == event][response]

  X_train, y_train = gdf_train[regressors], gdf_train[response]
  X_test, y_test = gdf_test[regressors], gdf_test[response]

  if scaling:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

  if visuals:
    sns.pairplot(gdf, x_vars=regressors, y_vars=response)
    sns.pairplot(gdf_train[regressors])

  # ensemble model
  y_pred = ensemble(X_train, y_train, X_test)
  y_rescaled = inverse_transform(y_pred)

  # predict and view test metrics
  mse = mean_squared_error(y_test, y_pred)
  rsq = rsquared(y_test, y_pred)
  mse_fullscale = mean_squared_error(y_fullscale, y_rescaled)
  rsq_fullscale = rsquared(y_fullscale, y_rescaled)
  print(f'{event}: MSE: {mse:.4f}, R^2: {rsq:.4f}')
  print(f'{event}: MSE (fullscale): {mse_fullscale:.4f}, R^2 (fullscale): {rsq_fullscale:.4f}')

  # scatter plot of observed vs predicted
  if visuals:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='k')
    ax.plot(y_test, y_test, color='r', label='test')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('XGBoost')

    fig, ax = plt.subplots()
    ax.scatter(y_fullscale, y_rescaled, color='k')
    ax.plot(y_fullscale, y_fullscale, color='r', label='test')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('XGBoost (rescaled)')

  predictions.append(y_rescaled)
  mses.append(mse_fullscale)
  rsqs.append(rsq_fullscale)

# gather results
results['predictions'] = np.concatenate(predictions)
rsqs = [rsq for rsq in rsqs if not np.isnan(rsq)]
mses = [mse for mse in mses if not np.isnan(mse)]

# %% ----Visualise results----
hist_kwargs = {'bins': 50, 'alpha': 0.5, 'color': 'lightgrey', 'edgecolor': 'k'}
plt.rcParams['font.family'] = 'serif'

fig, axs = plt.subplots(1, 3, figsize=(13, 3))
ax = axs[0]
ax.hist(mses, **hist_kwargs)
ax.set_xlabel('MSE')
ax.set_ylabel('Frequency')
ax.set_title('Mean squared error')

ax = axs[1]
q = .9
rsqs_clipped = np.clip(rsqs, np.quantile(rsqs, 0), np.quantile(rsqs, q))
ax.hist(rsqs_clipped, **hist_kwargs)
ax.set_xlabel('Pseudo-R²')
ax.set_ylabel('Frequency')
ax.set_title(f"Pseudo-R² (clipped at {q:.0%})")

ax = axs[2]
ax.scatter(results['y'], results['predictions'], color='k', s=.1, label='Predictions')
ax.plot(results['y'], results['y'], color='r', linewidth=.5, label='Observations')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.legend(loc='upper left')
ax.set_title('Predictions vs observations')

fig.suptitle('XGBoost and linear ensemble model', y=1.05)

#  calculate final RMSE and R2 
mse = mean_squared_error(results['y'], results['predictions'])
rsq = rsquared(results['y'], results['predictions'])

print(f'\nAveraged OOB scores: MSE: {mse:.4f}, R²: {rsq:.4f}')

# %%
from joblib import dump

modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'model.pkl')
with open(modelpath, "wb") as f:
    dump(model, f, protocol=5)

# %%
"""
----Full regressor options----
regressor_rename = {
    'landingSpeed': 'storm_speed',
    'totalPrec_total': 'precip',
    'elevation_mean': 'elevation',
    'bathymetry_mean': 'bathymetry',
    'era5_wind': 'wind',
    'era5_precip': 'precip',
    'era5_pressure': 'mslp',
    'wind': 'wind',
    'mslp': 'mslp',
    'continent': 'continent',
    'slope': 'slope',
    'landingWindMaxLocal2': 'wind',
    'landingPressure': 'mslp',
    'stormFrequency_mean': 'freq',
    'coastDist': 'dist_coast'
    }
"""




















