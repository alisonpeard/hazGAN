"""For development."""
# %%
import os
import yaml
from environs import Env
import hazGAN as hazzy

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
# %%
config['nconditions'] = len(metadata['labels'])
cgan = hazzy.conditional.compile_wgan(config)
cgan.fit(train, epochs=1, steps_per_epoch=10)
# brute force check length 
# should have a __len__ property after 1 epoch...
# %%