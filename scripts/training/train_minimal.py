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
    label_ratios={'pre':1/3, 7: 1/3, 20: 1/3},
    batch_size=128
    )

# %%
reload(hazzy)
tf.config.run_functions_eagerly(True) # to see pring statetements
config['nconditions'] = len(metadata['labels'])
cgan = hazzy.conditional.compile_wgan(config)
cgan.fit(train, epochs=1, steps_per_epoch=1)

# %%