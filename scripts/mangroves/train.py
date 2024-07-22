"""
To load model elsewhere, you can use the following snippet:
>>> from joblib import load
>>> modelpath = '/Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/model.pkl'
>>> with open(modelpath, "rb") as f:
>>>     model = load(f)


-----Input files-----
  - /Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/era5_and_slope_0.csv
-----Output files-----
  - /Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/model.pkl
"""
#%%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

bob_crs = 24346
# %%
gdf = gpd.read_file('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/data_with_slopes.gpkg')
df = pd.read_csv('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/data_with_era5.csv')
df['storm'] = df['storm'].str.capitalize()
# %%
final_df = gdf.merge(df, left_on=['center_centerLat', 'center_centerLon', 'stormName'], right_on=['lat', 'lon', 'storm'], how='left')
final_df = final_df.dropna()
final_df.to_csv('/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/final.csv', index=False)
# %%
path = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/final.csv'
df = pd.read_csv(path)
[*df.columns]
# %% ------Train a model------
# TODO: pressure, precipitation checks
regressors = [
    'slope',
    # 'landingWindMaxLocal2',
    # 'landingPressure',
    'era5_wind',
    'era5_pressure',
    # 'era5_precip',
    # 'stormFrequency_mean',
    # 'elevation_mean',
    # 'totalPrec_total',
    # 'bathymetry_mean',
    # 'landingSpeed',
    # 'coastDist'
    ]
response = ['intensity']
# %%
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_centerLon, df.center_centerLat)).set_crs(epsg=4326)

gdf = gdf.dropna()
X = gdf[regressors]
y = gdf[response]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
if True:
  import seaborn as sns

  sns.pairplot(gdf, x_vars=regressors, y_vars=response)
  sns.pairplot(X)
# %% 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %% train a random forest
for i in [1, 2, 5, 10, 20, 50, 100, 200]:
  model = RandomForestRegressor(n_estimators=i * len(regressors), random_state=42).fit(X_train, y_train)

  y_pred = model.predict(X_test)
  rmse = root_mean_squared_error(y_test, y_pred)
  rsq = r2_score(y_test, y_pred)

  print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}, nestimars factor: {i})')
# %%
# import simple mlp
from sklearn.neural_network import MLPRegressor
# %%
# try svm, xgboost, gaussian process regressors
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern

# %%
# model = XGBRegressor().fit(X_train, y_train)
# model = SVR().fit(X_train, y_train)
model = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()).fit(X_train, y_train)
# model = RandomForestRegressor().fit(X_train, y_train)
# model = MLPRegressor(hidden_layer_sizes=(100, 100)).fit(X_train, y_train)
# model = MLPRegressor(hidden_layer_sizes=(100, 100, 100)).fit(X_train, y_train)
# model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100)).fit(X_train, y_train)
# model = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100)).fit(X_train, y_train)
# model = MLPRegressor().fit(X_train, y_train)

# other variations of the gaussian process regressor
# model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel()).fit(X_train, y_train)
model = GaussianProcessRegressor(kernel=Matern() + WhiteKernel()).fit(X_train, y_train)


y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
rsq = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}, R^2: {rsq:.4f}')
# %%
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='k')
ax.plot(y_test, y_test, color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title(f'R^2 = {rsq:.4f}')
# %% plot the feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(model.feature_names_in_, importances[indices])
plt.xticks(rotation=45)

# %%
model.feature_names_in_
# %% pickle model to use later -- try other methods later
# Here you can replace pickle with joblib or cloudpickle
from joblib import dump
modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'model.pkl')
with open(modelpath, "wb") as f:
    dump(model, f, protocol=5)

# %%
