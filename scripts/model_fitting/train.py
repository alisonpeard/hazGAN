"""
Run in conda env tf_geo (tf 2.12.0, python 3.10)
Note, requires config to create new model too.

>>> new_gan = WGAN.WGAN(config)
>>> new_gan.generator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_generator_weights'))
>>> new_gan.critic.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_critic_weights'))

$ conda activate hazGAN
# python 2_train.py
$ tensorboard --logdir ./_logs
"""
# %%
import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
# tf.debugging.enable_check_numerics()

import wandb
from wandb.keras import WandbMetricsLogger
import hazGAN as hg


global rundir
global runname
global debug

plot_kwargs = {"bbox_inches": "tight", "dpi": 300}

# some static variables
data_source = "era5"
res = (18, 22)
cwd = os.getcwd()  # scripts directory
wd = os.path.join(cwd, "..", '..')  # hazGAN directory
datadir = os.path.join(wd, "..", f"{data_source}_data", f"res_{res[0]}x{res[1]}")  # keep data folder in parent directory
imdir = os.path.join(wd, "figures", "temp")
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])


def log_image_to_wandb(fig, name: str, dir: str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath, **plot_kwargs)
    wandb.log({name: wandb.Image(impath)})


def save_config(dir):
    configfile = open(os.path.join(dir, "config-defaults.yaml"), "w")
    configdict = {
        key: {"value": value} for key, value in wandb.config.as_dict().items()
    }
    yaml.dump(configdict, configfile)
    configfile.close()

# %%
def main(config):
    # load data
    data = hg.load_training(datadir, config.train_size, 'reflect', gumbel_marginals=config.gumbel)
    train_u = data['train_u']
    test_u = data['test_u']
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(config.batch_size)
    print(f"Training data shape: {train_u.shape}")
    
    # train test callbacks
    chi_score = hg.ChiScore({"train": next(iter(train)), "test": next(iter(test))},
                            frequency=config.chi_frequency, gumbel_margins=config.gumbel)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="chi_score_test",
        factor=0.1,
        patience=10,
        mode="min",
        verbose=1
        )

    # compile
    with tf.device("/gpu:0"):
        gan = getattr(hg, f"compile_{config.model}")(config, nchannels=2)
        gan.fit(
            train,
            epochs=config.nepochs,
            callbacks=[chi_score, WandbMetricsLogger(), hg.Visualiser(1, runname=runname), reduce_on_plateau]
        )

    # reproducibility
    gan.generator.save_weights(os.path.join(rundir, f"generator.weights.h5"))
    gan.critic.save_weights(os.path.join(rundir, f"critic.weights.h5"))
    save_config(rundir)

    # generate images to visualise some results
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    train_u = hg.unpad(train_u, paddings).numpy()
    test_u = hg.unpad(test_u, paddings).numpy()
    fake_u = hg.unpad(gan(nsamples=1000), paddings).numpy()
    fig = hg.plot_generated_marginals(fake_u, vmin=None, vmax=None, runname=runname)
    log_image_to_wandb(fig, f"generated_marginals", imdir)

    gan.critic.summary()
    gan.generator.summary()


# %% run this cell to train the model
if __name__ == "__main__":
    # parse arguments (for linux)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dry-run', dest="dry_run", action='store_true', default=False, help='Dry run')
    # args = parser.parse_args()
    # dry_run = args.dry_run
    dry_run = False

    # initialise wandb
    if dry_run:
        wandb.init(project="test", mode="disabled")
        wandb.config.update({'nepochs': 1, 'batch_size': 1, 'train_size': 1}, allow_val_change=True)
        runname = 'dry-run'
    else:
        wandb.init()  # saves snapshot of code as artifact
        runname = wandb.run.name
    rundir = os.path.join(wd, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)

    # set seed for reproductibility
    wandb.config["seed"] = np.random.randint(0, 1e6)
    tf.keras.utils.set_random_seed(wandb.config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()        # removes stochasticity from individual operations
    
    main(wandb.config)
# %% 
