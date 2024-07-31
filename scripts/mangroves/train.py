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
import matplotlib.pyplot as plt
import seaborn as sns
from MangroveDamage import MangroveDamageModel

visuals = False
# %%
def rsquared(y, yhat):
    """Psuedo R^2"""
    ymean = np.mean(y)
    y = np.where(y == ymean, np.nan, y)
    return sum((yhat - ymean)**2) / sum((y - ymean)**2)


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
    'landingWindMaxLocal2': 'wind'
    }
regressors = [value for value in regressor_rename.values()]

# %% load and process data
df = pd.read_csv(infile).rename(columns=regressor_rename)
events = list(set(df['landing']))

# %% ----Train model eventwise OOB---- 
from tqdm import tqdm

results = df[['landing', 'intensity']].rename(columns={'intensity': 'y'})
predictions = []
mses = []
rsqs = []
mse_fitted = []
rsq_fitted = []

# loop through events
for event in (pbar:=tqdm(events)):
  gdf_train = df[df['landing'] != event]
  gdf_test = df[df['landing'] == event]

  X_train, y_train = gdf_train[regressors], gdf_train[response]
  X_test, y_test = gdf_test[regressors], gdf_test[response]

  # ensemble model
  model = MangroveDamageModel()
  model.fit(X_train, y_train)
  y_fit = model.predict(X_train)
  y_pred = model.predict(X_test)

  # fit metrics
  mse_fit = mean_squared_error(y_train, y_fit)
  rsq_fit = rsquared(y_train, y_fit)
  print(f'{event}: fitted MSE: {mse_fit:.4f}, fitted R^2: {rsq_fit:.4f}')

  # predict and view test metrics
  mse = mean_squared_error(y_test, y_pred)
  rsq = rsquared(y_test, y_pred)
  print(f'{event}: MSE: {mse:.4f}, R^2: {rsq:.4f}')

  # store results
  predictions.append(y_pred)
  mse_fitted.append(mse_fit)
  rsq_fitted.append(rsq_fit)
  mses.append(mse)
  rsqs.append(rsq)
  

# gather results
results['predictions'] = np.concatenate(predictions)
rsqs = [rsq for rsq in rsqs if not np.isnan(rsq)]
mses = [mse for mse in mses if not np.isnan(mse)]
rsq_fitted = [rsq for rsq in rsq_fitted if not np.isnan(rsq)]
mse_fitted = [mse for mse in mse_fitted if not np.isnan(mse)]

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

# calculate fitted RMSE and R2
mse_fitted = np.mean(mse_fitted)
rsq_fitted = np.mean(rsq_fitted)
print(f'Fitted scores: MSE: {mse_fitted:.4f}, R²: {rsq_fitted:.4f}')

# %%
modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'damagemodel.pkl')
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




















