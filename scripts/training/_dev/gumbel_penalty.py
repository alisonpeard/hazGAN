# %%
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import hazGAN as hg

datadir = "/Users/alison/Documents/DPhil/paper1.nosync/training/res_18x22/"
data = hg.load_training(datadir, 1000, 'reflect', gumbel_marginals=True)
train = data['train_x']
train = tf.reshape(train,[1000, 18 * 22, 2])
# %%
i = 150
x = train[:, i, 0].ravel()
# %%
import matplotlib.pyplot as plt
plt.hist(x)
# %%
