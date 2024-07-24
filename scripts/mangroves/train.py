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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def transform(df: pd.DataFrame, X: list, y: str, test_size=0.1, random_state=42):
 # shift transform and scale
 df = df[[y] + X].dropna()
#  df[X] = df[X].apply(shift_positive, axis=0)
#  df = np.log(df)
 return(df)

def inverse_transform(y: pd.Series):
  return(y)
  # return np.exp(y)

def shift_positive(x: pd.Series):
  if x.min() <= 0:
    x += abs(x.min()) + 1
  return x

visuals = True

bob_crs = 24346
indir_mine = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'
indir_yus = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2'
infile1 = os.path.join(indir_mine, 'data_with_slopes.csv')
infile2 = os.path.join(indir_mine, 'final.csv')
infile3 = os.path.join(indir_mine, 'era5_and_slope_0.csv')

response = 'intensity'
regressor_rename = {
    # 'landingSpeed': 'storm_speed,
    # 'totalPrec_total': 'precip,
    # 'elevation_mean': 'elevation',
    # 'bathymetry_mean': 'bathymetry',
    'era5_wind': 'wind',
    'era5_pressure': 'mslp',
    # 'wind': 'wind',
    # 'mslp': 'mslp',
    # 'era5_precip': 'precip',
    # 'slope': 'slope',
    # 'landingWindMaxLocal2': 'wind',
    # 'landingPressure': 'mslp',
    # 'stormFrequency_mean': 'freq',
    # 'coastDist': 'dist_coast'
    }
regressors = [value for value in regressor_rename.values()]

# %% load and process data
scaler = StandardScaler()
df = pd.read_csv(infile2).rename(columns=regressor_rename)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)
gdf = transform(gdf, regressors, response)
X = gdf[regressors]
y = gdf[response]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

if visuals:
  sns.pairplot(gdf, x_vars=regressors, y_vars=response)
  sns.pairplot(X)
#
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
model = RandomForestRegressor(n_estimators=i * len(regressors), random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
rsq = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')

# plot observed vs predicted
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='k')
ax.plot(y_test, y_test, color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title(f'R^2 = {rsq:.4f}')

# plot feature importances
hist_kwargs = {'alpha': 0.5, 'color': 'lightblue', 'edgecolor': 'k'}
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots()
ax.bar(regressors, importances[indices], **hist_kwargs)
ax.set_xticks(ax.get_xticks(), labels=regressors, rotation=45, ha='right')
ax.set_ylabel('Importance')

# %% pickle that thing
modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'model.pkl')
with open(modelpath, "wb") as f:
    dump(model, f, protocol=5)

# scalerpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'scaler.pkl')
# with open(scalerpath, "wb") as f:
#     dump(scaler, f, protocol=5)

# %% ---- Here's some experiments with other model types ----
if False:
  # import simple mlp
  from sklearn.neural_network import MLPRegressor
  from sklearn.svm import SVR
  from xgboost import XGBRegressor
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern

  # Choose your fighter:
  model = SVR().fit(X_train, y_train)
  # model = XGBRegressor().fit(X_train, y_train)

  # # MLPs (increasing depth)
  # model = MLPRegressor().fit(X_train, y_train)
  # model = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, y_train)
  # model = MLPRegressor(hidden_layer_sizes=(100, 100, 100)).fit(X_train, y_train)
  # model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100)).fit(X_train, y_train)
  # model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100)).fit(X_train, y_train)

  # # Gaussian processes
  # model = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()).fit(X_train, y_train)
  # model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel()).fit(X_train, y_train)
  # model = GaussianProcessRegressor(kernel=Matern() + WhiteKernel()).fit(X_train, y_train)

  # predict and view test metrics
  y_pred = model.predict(X_test)
  rmse = root_mean_squared_error(y_test, y_pred)
  rsq = r2_score(y_test, y_pred)
  print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')

  # scatter plot of observed vs predicted
  fig, ax = plt.subplots()
  ax.scatter(y_test, y_pred, color='k')
  ax.plot(y_test, y_test, color='r', label='test')
  ax.set_xlabel('Observed')
  ax.set_ylabel('Predicted')

  #  plot the feature importances
  importances = model.feature_importances_
  indices = np.argsort(importances)[::-1]
  plt.bar(model.feature_names_in_, importances[indices])
  plt.xticks(rotation=45)

  # %%
