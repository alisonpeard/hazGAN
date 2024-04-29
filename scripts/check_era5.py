## %
import os
import pandas as pd
import matplotlib.pyplot as plt
# %%

path = "/Users/alison/Documents/DPhil/multivariate/era5_data/data_1979_2022.csv"
data = pd.read_csv(path)
data.describe()
# %%
data['time'] = pd.to_datetime(data['time'])

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].scatter(data['time'], data['u10'], s=.01, c='k')
axs[1].scatter(data['time'], data['msl'], s=.01, c='k')

axs[0].set_title('Daily maximum 10m wind')
axs[1].set_title('Daily minimum sea level pressure')
# %%
print(data.shape)
# %%
