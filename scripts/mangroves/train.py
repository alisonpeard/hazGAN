"""
To load model elsewhere, you can use the following snippet:
>>> from joblib import load
>>> modelpath = '/Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/model.pkl'
>>> with open(modelpath, "rb") as f:
>>>     model = load(f)

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
#%%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from joblib import dump # for pickling
import matplotlib.pyplot as plt
import seaborn as sns
from MangroveDamage import MangroveDamageModel

SCALING = True
IMDAA = False
VISUALS = False
AUGMENT = False # just shifts data, it can't find an underlying relationship that doesn't exist

# %%
def convert_ibtracs_vars(df):
    df = df.copy()
    def kmph_to_mps(x):
            return x * 1000 / 3600
    def mm_to_m(x):
        return x / 1000
    df['landingWindMaxLocal2'] = df['landingWindMaxLocal2'].apply(kmph_to_mps)
    df['totalPrec_total'] = df['totalPrec_total'].apply(mm_to_m)
    return df

def rsquared(y, yhat):
    """Psuedo R^2"""
    ymean = np.mean(y)
    y = np.where(y == ymean, np.nan, y)
    yhat = np.where(yhat == ymean, np.nan, yhat)
    sse = sum((y - yhat)**2)     # sum squared residuals
    sst = sum((y - ymean)**2)    # total sum of squares
    ssr = sum((yhat - ymean)**2) # sum of squares of regression
    return ssr / (ssr + sse) # Krueck (2020)
    # return ssr / sst # https://doi.org/10.1016/j.neunet.2009.07.002


def root_mean_squared_error(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))


bob_crs = 24346
indir_alison = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'
indir_yu = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v2'
infile0 = os.path.join(indir_yu, 'result/model/input_fixedCNTRY_rmOutlier.csv')
infile1 = os.path.join(indir_alison, 'data_with_slopes.csv')
infile2 = os.path.join(indir_alison, 'final.csv')
infile3 = os.path.join(indir_alison, 'era5_and_slope_0.csv')
infile4 = os.path.join(indir_alison, 'data_with_imdaa.csv')


infile = infile4
response = 'intensity'
regressor_rename = {
    # "imdaa_gust": "wind",
    # "imdaa_precip": "precip",
    # 'elevation_mean': 'elevation',
    # 'era5_wind': 'wind',
    # 'era5_precip': 'precip',
    # 'era5_pressure': 'mslp'
    # 'stormFrequency_mean': 'freq', # add this back in later
    # 'slope': 'slope',
    # # 'landingPressure': 'mslp'
    'totalPrec_total': 'precip',
    'landingWindMaxLocal2': 'wind'
    }
regressors = [value for value in regressor_rename.values()]

# %% load and process data
eventcol = 'stormName'
trainratio = 0.9
bootstrap = True
df = pd.read_csv(infile)

if IMDAA:
    df2 = pd.read_csv(infile0)
    df2['stormName'] = df2['stormName'].str.upper()
    df = pd.merge(df, df2,
                how='inner',
                left_on=['storm', 'lat', 'lon'],
                right_on=['stormName', 'center_centerLat', 'center_centerLon']
                )

# %%
df = convert_ibtracs_vars(df)
df = df.rename(columns=regressor_rename)
df = df.dropna(subset=regressors + [response])
events = list(set(df[eventcol]))
ntrain = int(len(events) * trainratio)

median = np.median(df[response]) # mangroves that are over 10% damaged
df[response] = (df[response] > median).astype(int) # binary problem
df[response].value_counts()

# %% ----Train model eventwise OOB---- 
# results = df[['landing', 'intensity']].rename(columns={'intensity': 'y'})
observations = []
predictions = []
rmses = []
rsqs = []
rmse_fitted = []
rsq_fitted = []

# loop through events
for i in range(100):
  train_events = np.random.choice(events, ntrain, replace=bootstrap)
  df_train = df[df[eventcol].isin(train_events)]
  df_test = df[~df[eventcol].isin(train_events)]
  print("train : test", len(df_train), ":", len(df_test))

  X_train, y_train = df_train[regressors], df_train[response]
  X_test, y_test = df_test[regressors], df_test[response]

  # ensemble model
  model = MangroveDamageModel(scaling=SCALING)
  model.fit(X_train, y_train)
  y_fit = model.predict(X_train)
  y_pred = model.predict(X_test)

  # fit metrics
  rmse_fit = root_mean_squared_error(y_train, y_fit)
  rsq_fit = rsquared(y_train, y_fit)

  # predict and view test metrics
  rmse = root_mean_squared_error(y_test, y_pred)
  rsq = rsquared(y_test, y_pred)

  print(f'Bootstrap {i}: fitted rmse:  {rmse_fit:.4f}, fitted R^2: {rsq_fit:.4f}')
  print(f' ---- rmse: {rmse:.4f}, R^2: {rsq:.4f}')

  # store results
  observations.append(y_test)
  predictions.append(y_pred)
  rmse_fitted.append(rmse_fit)
  rsq_fitted.append(rsq_fit)
  rmses.append(rmse)
  rsqs.append(rsq)
  

# gather results
observations = np.concatenate(observations)
predictions = np.concatenate(predictions)
rsqs = [rsq for rsq in rsqs if not np.isnan(rsq)]
mses = [rmse for rmse in rmses if not np.isnan(rmse)]
rsq_fitted = [rsq for rsq in rsq_fitted if not np.isnan(rsq)]
rmse_fitted = [rmse for rmse in rmse_fitted if not np.isnan(rmse)]

# %% ----Visualise results----
hist_kwargs = {'bins': 50, 'alpha': 0.5, 'color': 'lightgrey', 'edgecolor': 'k'}
plt.rcParams['font.family'] = 'serif'

fig, axs = plt.subplots(1, 4, figsize=(15, 3))
ax = axs[0]
ax.hist(rmses, **hist_kwargs)
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
ax.scatter(observations, predictions, color='k', s=.1, label='Predictions')
ax.plot(observations, observations, color='r', linewidth=.5, label='Observations')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.legend(loc='upper left')
ax.set_title('Predictions vs observations')

ax = axs[3]
ax.hist(observations, **hist_kwargs)
ax.set_title("Distribution of observations")

fig.suptitle('XGBoost and linear ensemble model', y=1.05)

# calculate fitted RMSE and R2
rmse_fitted = np.mean(rmse_fitted)
rsq_fitted = np.mean(rsq_fitted)
print(f'Fitted scores: rmse:  {rmse_fitted:.4f}, R²: {rsq_fitted:.4f}')

#  calculate final RMSE and R2 
rmse = root_mean_squared_error(observations, predictions)
rsq = rsquared(observations, predictions)
print(f'Averaged OOB scores: RMSE: {rmse:.4f}, R²: {rsq:.4f}')

# %% fit model on all data and save
X, y = df[regressors], df[response]
model.fit(X, y)
model.set_metrics({'rmse': rmse, 'rsq': rsq, 'rmse_fitted': rmse_fitted, 'rsq_fitted': rsq_fitted})
modelpath = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync', 'results', 'mangroves', 'damagemodel.pkl')
with open(modelpath, "wb") as f:
    dump(model, f, protocol=5)

# %%
median = np.median(observations)
y_binary = np.where(observations > .5, 1, 0)
predictions_binary = np.where(predictions > .5, 1, 0)

confusion_matrix = pd.crosstab(y_binary, predictions_binary, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
#%%
# calculate precision, recall, and csi
TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
TN = confusion_matrix[0][0]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
csi = TP / (TP + FN + FP)

print(f'Fitted scores: rmse:  {rmse_fitted:.4f}, R²: {rsq_fitted:.4f}')
print(f'Averaged OOB scores: RMSE: {rmse:.4f}, R²: {rsq:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, CSI: {csi:.4f}')
# print(model.base.coef_)
# %%