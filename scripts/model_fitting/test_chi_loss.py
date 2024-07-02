# %%
import hazGAN as hg
import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
# tf.debugging.enable_check_numerics() # check for NaN/Inf in tensors

import wandb
from wandb.keras import WandbMetricsLogger

# %%
data_source = "era5"
res = (18, 22)
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

wandb.init()
config = wandb.config


wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')  # hazGAN directory
datadir = os.path.join(wd, 'training', f"res_{res[0]}x{res[1]}")  # keep data folder in parent directory


data = hg.load_training(datadir, config.train_size, 'reflect', gumbel_marginals=config.gumbel)
train_u = data['train_u']
test_u = data['test_u']
train = tf.data.Dataset.from_tensor_slices(train_u).batch(config.batch_size)
test = tf.data.Dataset.from_tensor_slices(test_u).batch(config.batch_size)
# %%
from hazGAN import compile_wgan, chi_loss

real = next(iter(train))
fake = tf.random.uniform(real.shape)
with tf.device('/gpu:0'):
    res = chi_loss(real, fake)

print(res)
# %%
