"""
Sandbox 06-12-2024
 - Remove outliers from pretraining data and process 60,000 samples for DCGAN
"""
# %%
import os
from environs import Env
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

env = Env()
env.read_env(recurse=True)
datadir = env.str("TRAINDIR")

# %% ----PRETRAINING PROCESSING----
ds = xr.open_dataset(os.path.join(datadir, 'data_pretrain.nc'))
ds = ds.sel(channel='u10')
ds['maxwind'] = ds['anomaly'].max(dim=['lon', 'lat'])
sorting = ds['maxwind'].argsort()

# %% plot 100 largest sample
fig, axs = plt.subplots(20, 20, figsize=(20, 20), sharex=True, sharey=True)

for i, ax in enumerate(axs.flat):
    ds['anomaly'].isel(time=sorting[-i]).plot(
        ax=ax, vmin=-40, vmax=40, cmap="Spectral_r", add_colorbar=False)
    ax.axis('off')
    ax.set_title('')
fig.suptitle('400 largest samples')

# %%
!say done
# %%
fig, axs = plt.subplots(10, 10, figsize=(20, 20), sharex=True, sharey=True)
for i, ax in enumerate(axs.flat):
    ds['anomaly'].isel(time=sorting[i]).plot(
        ax=ax, vmin=-10, vmax=10, cmap="Spectral_r", add_colorbar=False)
    ax.axis('off')
    ax.set_title('')
fig.suptitle('100 smallest samples')

print(ds.uniform.data.shape)
# %% get into 28 x 28 [-1, 1] format for training in MNIST-DCGAN example
x = ds.uniform.data.transpose(2, 0, 1)
x = x[:, ::-1, :, np.newaxis]
x = (x - 0.5 * x.max()) / (0.5 * x.max())
x = tf.image.resize(x, (28, 28)).numpy().astype('float32')
print(x.shape)
plt.imshow(x[0, ..., 0], cmap='Spectral_r')
# np.savez(os.path.join(datadir, "data_filtered_60000.npz"), x=x)

# %% ----END OF PRETRAINING PROCESSING----

ds = xr.open_dataset(os.path.join(datadir, 'data.nc'))
# 1228 samples, 18 x 22

# %%

train = ds.sel(channel='u10')['uniform'].data[..., np.newaxis]
train = tf.image.resize(train, (28,28)).numpy().astype('float32')
train.shape # (1228, 28, 28)
print("min: {}, max: {}".format(train.min(), train.max()))
# %%
train = (train - .5) / .5
print("min: {}, max: {}".format(train.min(), train.max()))
# %%
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(train_images.shape) # (60000, 28, 28)
print(train_images.max()) # 255
# %%
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


