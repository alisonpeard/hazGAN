# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import genpareto
import hazGAN as hg
from importlib import reload

channels = ["U10", "MSLP"]
wd = "/Users/alison/Documents/DPhil/paper1.nosync"
datadir = os.path.join(wd, "training", '18x22')
# %%
i, j, c = 0, 0, 0
# u, x, m, z, params = hg.load_training(datadir, 1000)
training = hg.load_training(datadir, 1000)

u = training['train_u'][:, i, j, c]
x = training['train_x'][:, i, j, c]
m = training['train_m'][:, i, j, c]
z = training['train_z'][0]
params = training['params'][i, j, :, c]
shape, loc, scale = params

# %% plot data, GPD fit, and transformed data
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].hist(x, bins=20, alpha=0.5, edgecolor='k', label=channels[c], density=True);
xmin = genpareto.ppf(0, shape, loc, scale)
xmax = genpareto.ppf(1, shape, loc, scale)
xs = np.linspace(xmin, xmax, 1000)
axs[1].hist(x[x > loc], bins=20, alpha=0.5, edgecolor='k', label=channels[c], density=True);
axs[1].plot(xs, genpareto.pdf(xs, shape, loc, scale), label="GPD fit", color='red', linestyle='dashed')
axs[1].legend()
axs[2].hist(u, bins=20, alpha=0.5, edgecolor='k', label="u", density=True)

axs[0].set_title(channels[c])
axs[1].set_title(f"POT excesses {channels[c]}")
axs[2].set_title(f"Transformed {channels[c]}")
plt.show()
# %% plot the inverse transforms to make sure they look sensible
x_inv_empirical = hg.empirical_quantile(u, x, u, None)
x_inv = hg.empirical_quantile(u, x, u, params)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
sns.jointplot(x=u, y=x, height=3)
sns.jointplot(x=u, y=x_inv_empirical, height=3)
sns.jointplot(x=u, y=x_inv, height=3)

# %% visualise the inverse transformed data
i, c = 10, 1
# u, x, m, z, params = hg.load_training(datadir, 1000, zero_pad=False, numpy=True)
# u = u[0]
# x = x[0]
# m = m[0]
# z = z[0]

training = hg.load_training(datadir, 1000, padding_mode=None)

u = training['train_u'].numpy()
x = training['train_x'].numpy()
m = training['train_m'].numpy()
z = training['train_z']
params = training['params']
shape, loc, scale = params[..., 0, :], params[..., 1, :], params[..., 2, :]


params = params
x_inv = hg.POT.inv_probability_integral_transform(u, x, u, params)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
im = axs[0].imshow(x[i, ..., c], cmap='viridis')
plt.colorbar(im, orientation='horizontal')
im = axs[1].imshow(u[i, ..., c], cmap='viridis')
plt.colorbar(im, orientation='horizontal')
im = axs[2].imshow(x_inv[i, ..., c], cmap='viridis')
plt.colorbar(im, orientation='horizontal')
axs[0].set_title(f"{channels[c]}")
axs[1].set_title(f"transformed to uniform")
axs[2].set_title(f"inverse transformed")
# %%
