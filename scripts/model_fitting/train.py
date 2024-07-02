"""
-----Requirements-----
    - env: hazGAN
    - GAN configuratino: config-defaults.yaml
    - data: era5_data/res_18x22/data.nc

-----Output-----
    - saved_models: generator.weights.h5, critic.weights.h5

-----To use saved weights-----
>>> new_gan = WGAN.WGAN(config)
>>> new_gan.generator.load_weights(os.path.join(wd, 'saved_models', runname, 'generator_weights'))
>>> new_gan.critic.load_weights(os.path.join(wd, 'saved_models', runname, 'critic_weights'))
or
>>> new_gan.load_weights(os.path.join(rundir, 'checkpoint.weights.h5'))

-----Linux cluster examples-----
$srun -p Short --pty python train.py --dry-run --cluster # dry run
$ 
"""
# %%
import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()

# debugging ops
# tf.debugging.set_log_device_placement(True)
# tf.debugging.enable_check_numerics() # check for NaN/Inf in tensors

import wandb
from wandb.keras import WandbMetricsLogger
import hazGAN as hg


global rundir
global runname
global force_cpu

plot_kwargs = {"bbox_inches": "tight", "dpi": 300}

# some static variables
data_source = "era5"
res = (18, 22)
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


def config_tf_devices():
    """Use GPU if available"""
    gpus = tf.config.list_logical_devices("GPU")
    gpu_names = [x.name for x in gpus]
    if (len(gpu_names) == 0) or force_cpu:
        print("No GPU found, using CPU")
        cpus = tf.config.list_logical_devices("CPU")
        cpu_names = [x.name for x in cpus]
        return cpu_names[0]
    else:
        print(f"Using GPU: {gpu_names[0]}")
        return gpu_names[0]

# %% minimal example
wandb.init(project='test', mode='disabled')
config = wandb.config
gan = hg.compile_wgan(config, nchannels=2)
# %%
def main(config):
    # load data
    data = hg.load_training(datadir, config.train_size, 'reflect', gumbel_marginals=config.gumbel)
    train_u = data['train_u']
    test_u = data['test_u']
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(config.batch_size)
    print(f"Training data shape: {train_u.shape}")
    
    # define callbacks
    visualiser = hg.Visualiser(1, runname=runname)
    
    chi_score = hg.ChiScore({
            "train": next(iter(train)),
            "test": next(iter(test))
            },
        frequency=config.chi_frequency,
        gumbel_margins=config.gumbel
        )
    
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="generator_loss",
        factor=config.lr_factor,
        patience=config.lr_patience,
        mode="min",
        verbose=1
        )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(rundir, "checkpoint.weights.h5"),
        monitor="chi_score_test",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
        )

    # compile
    with tf.device(device):
        gan = getattr(hg, f"compile_{config.model}")(config, nchannels=2)
        gan.fit(
            train,
            epochs=config.nepochs,
            callbacks=[
                WandbMetricsLogger(),
                # visualiser,
                checkpoint
                ]
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


# %% run this cell to train the model
if __name__ == "__main__":
    # parse arguments (for linux)
    # if sys.__stdin__.isatty():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', '-d', dest="dry_run", action='store_true', default=False, help='Dry run')
    parser.add_argument('--cluster', '-c', dest="cluster", action='store_true', default=False, help='Running on cluster')
    parser.add_argument('--force-cpu', '-f', dest="force_cpu", action='store_true', default=False, help='Force use CPU (for debugging)')
    args = parser.parse_args()
    dry_run = args.dry_run
    cluster = args.cluster
    force_cpu = args.force_cpu
    # else:
    #     dry_run = True
    #     cluster = False
    #     force_cpu = False

    # setup device
    device = config_tf_devices()

    # set up directories
    if cluster:
        wd = os.path.join('/soge-home', 'projects', 'mistral', 'alison', 'hazGAN')
    else:
        wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')  # hazGAN directory

    datadir = os.path.join(wd, 'training', f"res_{res[0]}x{res[1]}")  # keep data folder in parent directory
    print(f"Loading data from {datadir}")
    imdir = os.path.join(wd, "figures", "temp")

    # initialise wandb
    if dry_run:
        print("Starting dry run")
        wandb.init(project="test", mode="disabled")
        wandb.config.update({'nepochs': 1, 'batch_size': 1, 'train_size': 1}, allow_val_change=True)
        runname = 'dry-run'
    else:
        wandb.init(project="hazGAN", allow_val_change=True)  # saves snapshot of code as artifact
        runname = wandb.run.name
    rundir = os.path.join(wd, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)

    # set seed for reproductibility
    wandb.config["seed"] = np.random.randint(0, 1e6)
    tf.keras.utils.set_random_seed(wandb.config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()        # removes stochasticity from individual operations
    
    main(wandb.config)
# %% 
