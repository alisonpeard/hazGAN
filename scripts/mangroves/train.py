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
scaling = True
visuals = False

def transform(df: pd.DataFrame, X: list, y: str):
 # shift transform and scale
 if scaling:
  df = df[[y] + X].dropna()
  df[X] = df[X].apply(shift_positive, axis=0)
  df = np.log(df)
 return(df)


def inverse_transform(y: pd.Series):
  if scaling:
    y = np.exp(y)
  return(y)


def shift_positive(x: pd.Series):
  if x.min() <= 0:
    x += abs(x.min()) + 1
  return x


def rsquared(y, yhat):
    return sum((yhat - np.mean(y))**2) / sum((y - np.mean(y))**2)


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
    # 'landingSpeed': 'storm_speed',
    'totalPrec_total': 'precip',
    # 'elevation_mean': 'elevation',
    # 'bathymetry_mean': 'bathymetry',
    # 'era5_wind': 'wind',
    # 'era5_precip': 'precip',
    # 'era5_pressure': 'mslp',
    # 'wind': 'wind',
    # 'mslp': 'mslp',
    # 'continent': 'continent',
    'slope': 'slope',
    'landingWindMaxLocal2': 'wind',
    # 'landingPressure': 'mslp',
    # 'stormFrequency_mean': 'freq',
    # 'coastDist': 'dist_coast'
    }
regressors = [value for value in regressor_rename.values()]
scaler = StandardScaler()

# %% load and process data
df = pd.read_csv(infile).rename(columns=regressor_rename)
events = [*df['landing'].unique()]

df_transformed = transform(df, regressors, response)
df_transformed[['landing', 'center_centerLon', 'center_centerLat']] = df[['landing', 'center_centerLon', 'center_centerLat']]
df = df_transformed
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
# %%
from tqdm import tqdm
from xgboost import XGBRegressor

results = gdf[['landing', 'intensity']].rename(columns={'intensity': 'y'})
predictions = []
mses = []
rsqs = []
for event in tqdm(events):
  gdf_train = gdf[gdf['landing'] != event]
  gdf_test = gdf[gdf['landing'] == event]
  X_train, y_train = gdf_train[regressors], gdf_train[response]
  X_test, y_test = gdf_test[regressors], gdf_test[response]

  if scaling:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

  if visuals:
    sns.pairplot(gdf, x_vars=regressors, y_vars=response)
    sns.pairplot(gdf_train[regressors])

  model = XGBRegressor().fit(X_train, y_train)

  # predict and view test metrics
  xgb_fit = model.predict(X_train)
  xgb_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, xgb_pred)
  rsq = rsquared_res(y_test, xgb_pred)
  print(f'MSE: {mse:.4f}, R^2: {rsq:.4f}')

  # scatter plot of observed vs predicted
  if visuals:
    fig, ax = plt.subplots()
    ax.scatter(y_test, xgb_pred, color='k')
    ax.plot(y_test, y_test, color='r', label='test')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('XGBoost')

  predictions.append(xgb_pred)
  mses.append(mse)
  rsqs.append(rsq)
results['predictions'] = np.concatenate(predictions)
# %%
hist_kwargs = {'bins': 50, 'alpha': 0.5, 'color': 'lightgrey', 'edgecolor': 'k'}
plt.rcParams['font.family'] = 'serif'

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
ax = axs[0]
ax.hist(mses, **hist_kwargs)
ax.set_xlabel('MSE')
ax.set_ylabel('Frequency')

ax = axs[1]
ax.hist(rsqs, **hist_kwargs)
ax.set_xlabel('R^2')
ax.set_ylabel('Frequency')

ax = axs[2]
ax.scatter(results['y'], results['predictions'], color='k')
ax.plot(results['y'], results['y'], color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('XGBoost')
# %% calculate final RMSE and R^2
rmse = root_mean_squared_error(results['y'], results['predictions'])
rsq = r2_score(results['y'], results['predictions'])

print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')
# %%
r2 = r2(results['y'], results['predictions'])
r2res = r2_res(results['y'], results['predictions'])
mse = mse(results['y'], results['predictions'])
mape = mape(results['y'], results['predictions'])

print(f'R^2: {r2:.4f}, R^2_res: {r2res:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}')

# %%



























# %% ----OLD STUFF----
# %% train a random forest for different numbers of trees
results = {'MSE': [], 'RMSE': [], 'R^2': [], 'K': [], 'K * npredictors': []}
for i in [1, 2, 5, 10, 20, 50, 100, 200]:
  model = RandomForestRegressor(n_estimators=i * len(regressors), random_state=42).fit(X_train, y_train)

  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  rmse = root_mean_squared_error(y_test, y_pred)
  rsq = r2_score(y_test, y_pred)

  results['MSE'].append(mse)
  results['RMSE'].append(rmse)
  results['R^2'].append(rsq)
  results['K'].append(i)
  results['K * npredictors'].append(i * len(regressors))

results = pd.DataFrame.from_dict(results)
results

# %% fit final model 
k = results.set_index('K').idxmax()['R^2']
model = RandomForestRegressor(n_estimators=i * len(regressors),
                              random_state=42).fit(X_train, y_train)
rf_fit = model.predict(X_train)
rf_pred = model.predict(X_test)
mse = mean_squared_error(y_test, rf_pred)
rmse = root_mean_squared_error(y_test, rf_pred)
rsq = r2_score(y_test, rf_pred)
print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')

# plot observed vs predicted
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
ax = axs[0]
ax.scatter(y_test, y_pred, color='k')
ax.plot(y_test, y_test, color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title(f'RF R^2 = {rsq:.4f}')

# plot feature importances
hist_kwargs = {'alpha': 0.5, 'color': 'lightblue', 'edgecolor': 'k'}
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
ax = axs[1]
ax.bar(regressors, importances[indices], **hist_kwargs)
ax.set_xticks(ax.get_xticks(), labels=regressors, rotation=45, ha='right')
ax.set_ylabel('Importance')

# %% pickle that thing
if False:
    modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'model.pkl')
    with open(modelpath, "wb") as f:
        dump(model, f, protocol=5)

    scalerpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'scaler.pkl')
    with open(scalerpath, "wb") as f:
        dump(scaler, f, protocol=5)

# %% ---- Here's some experiments with other model types ----
if True:
  from xgboost import XGBRegressor

  model = XGBRegressor().fit(X_train, y_train)

  # predict and view test metrics
  xgb_fit = model.predict(X_train)
  xgb_pred = model.predict(X_test)
  rmse = root_mean_squared_error(y_test, xgb_pred)
  rsq = r2_score(y_test, xgb_pred)
  print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')

  # scatter plot of observed vs predicted
  fig, ax = plt.subplots()
  ax.scatter(y_test, xgb_pred, color='k')
  ax.plot(y_test, y_test, color='r', label='test')
  ax.set_xlabel('Observed')
  ax.set_ylabel('Predicted')
  ax.set_title('XGBoost')

# %%  ensemble the two of them 
from sklearn.linear_model import LinearRegression

X_train_ensemble = np.column_stack([rf_fit, xgb_fit])
X_test_ensemble = np.column_stack([rf_pred, xgb_pred])

model = LinearRegression().fit(X_train_ensemble, y_train)
ens_pred = model.predict(X_test_ensemble)

rmse = root_mean_squared_error(y_test, ens_pred)
rsq = r2_score(y_test, ens_pred)
print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')

# scatter plot of observed vs predicted
fig, ax = plt.subplots()
ax.scatter(y_test, ens_pred, color='k')
ax.plot(y_test, y_test, color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Linear Ensemble')
# %%
