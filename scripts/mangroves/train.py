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
path = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine/era5_and_slope_0.csv'
df = pd.read_csv(path)
[*df.columns]
# %% ------Train a model------
# TODO: pressure, precipitation checks
regressors = [
    # 'slope',
    # 'landingWindMaxLocal2',
    # 'landingPressure'
    'wind',
    'mslp',
    # 'stormFrequency_mean',
    # 'elevation_mean',
    # 'totalPrec_total',
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
# %% train a random forest
model = RandomForestRegressor(n_estimators=100 * len(regressors), random_state=42).fit(X_train, y_train)

y_pred = model.predict(X_test)
root_mean_squared_error(y_test, y_pred)
# %%

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='k')
ax.plot(y_test, y_test, color='r', label='test')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title(f'R^2 = {r2_score(y_test, y_pred):.4f}')
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
