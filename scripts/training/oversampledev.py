# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# /Users/alison/Documents/DPhil/paper1.nosync/training
wd = os.path.join("/Users", "alison", "Documents", "DPhil", "paper1.nosync")
data = xr.open_dataset(os.path.join(wd, "training", "18x22", "data.nc"))
data['maxima'] = data.isel(channel=0).anomaly.max(dim=['lon', 'lat'])
# %%

bins = [0, 15, 100]
hist_kws = {'color': 'lightgrey', 'edgecolor': 'black', 'alpha': 0.7,
            'bins': bins}
plt.hist(data['maxima'].values, **hist_kws)
plt.show()

# %%  give each a class
def classify(data, var='maxima', bins=bins):
    data['class'] = 0 * data[var]
    data['class'] = data['class'] + bins[0] * (data[var] <= bins[0])
    data['class'] = data['class'] + bins[-1] * (data[var] > bins[-1])
    for bin0, bin1 in zip(bins, bins[1:]):
        data['class'] = data['class'] + bin0 * (
            (data[var] > bin0) & 
            (data[var] <= bin1)
            )
    return data['class']

# modify classify so that it includes all values below the first bin
data['class'] = classify(data)
counts, bins = np.histogram(data['maxima'].values, bins=bins)
print(bins, counts)

# %% make sampling algorithm
true_ratio = counts[0] / counts[1]

# %%
bin_indices = {bin: data.where(data['class'] == bin, drop=True).time for bin in np.unique(data['class'].values)}
subclass = data.sel(time=bin_indices[15])

# %%
# subclass.resample(time='D').mean().plot(x='time')
# https://github.com/pydata/xarray/blob/main/xarray/groupers.py#L143-L150
from xarray.groupers import Resampler, BinGrouper

class BalanceResampler(Resampler):
    pass

# %%
