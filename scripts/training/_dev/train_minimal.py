"""For development and testing."""
# %%
import os
import yaml
from importlib import reload
from environs import Env
import hazGAN as hazzy
import tensorflow as tf

env = Env()
env.read_env(recurse=True)

workdir = os.path.dirname(__file__)
datadir = env.str('TRAINDIR')

with open(os.path.join(workdir, "config-defaults.yaml"), 'r') as stream:
    config = yaml.safe_load(stream)
    config = {key: value['value'] for key, value in config.items()}


train, valid, metadata = hazzy.load_data(
    datadir,
    label_ratios=config['label_ratios'],
    batch_size=config['batch_size']
    )

# %%
reload(hazzy)
tf.config.run_functions_eagerly(True) # to see pring statetements
config['nconditions'] = len(metadata['labels'])
cgan = hazzy.conditional.compile_wgan(config)
cgan.fit(train, epochs=1, steps_per_epoch=1)

# %% EDA (move later)
import os
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt

env = Env()
env.read_env(recurse=True)
dir = env.str('TRAINDIR')

data = xr.open_dataset(os.path.join(dir, 'data.nc'))
data['maxwind'] = data['anomaly'].isel(channel=0).max(dim=['lat', 'lon'])
maxwind = data['maxwind'].data
plt.hist(maxwind)
# %%
data  = data.where(data['maxwind'] > 20, drop=True)
maxwind = data['maxwind'].data
plt.hist(maxwind) # this is distribution of maxwinds above 20 mps

# %% sample from this distribution
import numpy as np
sample = np.random.choice(maxwind, 100)
plt.hist(sample)

# %%
