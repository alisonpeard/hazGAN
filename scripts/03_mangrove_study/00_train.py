"""
Train and plot 2-d logistic regression model for mangrove damage.

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
from tqdm import tqdm
from joblib import dump # for pickling
import matplotlib.pyplot as plt
import seaborn as sns
from utils.MangroveDamage import MangroveDamageModel

LOCAL_CRS = 24346
SCALING = True
VISUALS = False
THRESHOLD = 0.2 # mangroves being 20% damaged or more

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


infile = "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN-data/mangroves/data-from-yu_v2.csv"

response = 'intensity'
regressor_rename = {
    'landingWindMaxLocal2': 'wind',
    'totalPrec_total': 'precip'
    }

regressors = [value for value in regressor_rename.values()]

# %% load and process data
eventcol = 'stormName'
trainratio = 0.9
bootstrap = False

if __name__ == "__main__":
    df = pd.read_csv(infile)

    # %%
    df = convert_ibtracs_vars(df)
    df = df.rename(columns=regressor_rename)
    df = df.dropna(subset=regressors + [response])
    events = list(set(df[eventcol]))
    ntrain = int(len(events) * trainratio)

    df[response] = (df[response] >= THRESHOLD).astype(int) # binary problem
    df[response].value_counts()

    # %% ----Train model eventwise OOB---- 
    # results = df[['landing', 'intensity']].rename(columns={'intensity': 'y'})
    observations = []
    predictions = []
    rmses = []
    rsqs = []
    rmse_fitted = []
    rsq_fitted = []
    winds = []
    precips = []

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
        winds.append(df_test['wind'])
        precips.append(df_test['precip'])
  

    # gather results
    observations = np.concatenate(observations)
    predictions = np.concatenate(predictions)
    winds = np.concatenate(winds)
    precips = np.concatenate(precips)
    rsqs = [rsq for rsq in rsqs if not np.isnan(rsq)]
    mses = [rmse for rmse in rmses if not np.isnan(rmse)]
    rsq_fitted = [rsq for rsq in rsq_fitted if not np.isnan(rsq)]
    rmse_fitted = [rmse for rmse in rmse_fitted if not np.isnan(rmse)]

    # %% ----Visualise results----
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
    modelpath = os.path.join("/Users/alison/Documents/DPhil/paper1.nosync/hazGAN-data/mangroves", 'damagemodel.pkl')
    with open(modelpath, "wb") as f:
        dump(model, f, protocol=5)

    # %% ----Get coefficients and significance----
    coefs = model.base.coef_
    coefs

    # %% ----Make a 2D version of this plot----
    from matplotlib.ticker import PercentFormatter

    # set font to Helvetica
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set frame on for top and right axes
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True

    # set fontsize to 18
    plt.rcParams['font.size'] = 12

    windmax, precipmax = np.max(X, axis=0)
    windmin, precipmin = np.min(X, axis=0)
    windregular = np.linspace(windmin, windmax, 100)
    precipregular = np.linspace(precipmin, precipmax, 100)

    windregular, precipregular = np.meshgrid(windregular, precipregular)
    regressors_2d = np.column_stack((windregular.flatten(), precipregular.flatten()))
    
    preds = model.predict(regressors_2d)
    preds = preds.reshape((100, 100))
    winds = regressors_2d[:, 0].reshape((100, 100))
    precips = regressors_2d[:, 1].reshape((100, 100))

    if True: # debugging
        # Check a corner point
        print("\nBottom-left corner:")
        print(f"Wind: {windregular[0,0]}, Precip: {precipregular[0,0]}")
        print(f"Regressors: {regressors_2d[0, 0], regressors_2d[0, 1]}")
        print(f"Prediction: {preds[0,0]}")

        # Manually predict this same point
        manual_pred = model.predict(np.array([[windregular[0,0], precipregular[0,0]]]))
        print(f"Manual prediction: {manual_pred}")

        # check another corner point
        print("\nTop-right corner:")
        print(f"Wind: {windregular[-1,-1]}, Precip: {precipregular[-1,-1]}")
        print(f"Regressors: {regressors_2d[-1,0], regressors_2d[-1,1]}")
        print(f"Prediction: {preds[-1,-1]}")
        manual_pred = model.predict(np.array([[windregular[-1,-1], precipregular[-1,-1]]]))
        print(f"Manual prediction: {manual_pred}")

        # check another corner point
        print("\nBottom-right corner:")
        print(f"Wind: {windregular[-1,0]}, Precip: {precipregular[-1,0]}")
        print(f"Prediction: {preds[-1,0]}")
        manual_pred = model.predict(np.array([[windregular[-    1,0], precipregular[-1,0]]]))
        print(f"Manual prediction: {manual_pred}")

        # check another corner point
        print("\nTop-left corner:")
        print(f"Wind: {windregular[0,-1]}, Precip: {precipregular[0,-1]}")
        print(f"Prediction: {preds[0,-1]}")
        manual_pred = model.predict(np.array([[windregular[0,-1], precipregular[0,-1]]]))
        print(f"Manual prediction: {manual_pred}")

    # %%
    preds   *= 100   # convert to percentage
    precips *= 1000  # convert to mm

    vmin = np.round(preds.min(), -1)
    vmax = np.round(preds.max(), -1)
    LEVELS = np.arange(vmin, vmax + 6, 2.5)

    fig = plt.figure(figsize=(4, 3))
    im = plt.contourf(precips, winds, preds, levels=LEVELS, cmap='YlOrRd', origin='lower')
    plt.contour(precips, winds, preds, levels=LEVELS, colors='k', origin='lower',
                linewidths=0.05)
    plt.ylabel(r'Wind speed (ms$^{-1}$)')
    plt.xlabel('Precipitation (mm)')

    plt.colorbar(im, label='Damage probability', format=PercentFormatter(100, 0),
                 extend="both", ticks=LEVELS[::4]
                 )
    plt.tight_layout()
    fig.savefig(os.path.join("/Users/alison/Documents/DPhil/paper1.nosync/hazGAN-data/", "figures", "logistic_model.pdf"), transparent=True, dpi=300)
    # %%
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
    print(model.base.coef_)
    # %%