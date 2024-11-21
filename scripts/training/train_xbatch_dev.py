# %% test script
import os
import yaml
import hazGAN as hazzy
from environs import Env

# %%
def read_config(dir=os.getcwd()):
    with open(os.path.join(dir, 'config-defaults.yaml')) as stream:
        config = yaml.safe_load(stream)

    config = {k: v['value'] for k, v in config.items()}
    return config

config = read_config()
config

#%%
env = Env()
env.read_env(recurse=True)
datadir = env.str("TRAINDIR")
BATCH_SIZE = 16

train = hazzy.load_training(datadir, BATCH_SIZE)
train
#%% compile


gan = hazzy.compile_wgan(config, nchannels=2)


# %%
print("\nStarting training...")
history = gan.fit(train, epochs=1)

# %%
gan.cleanup() #?
#%%