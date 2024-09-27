"""
Training script for hazGAN.

This script trains a GAN model on ERA5 data to generate
synthetic wind fields. The script uses the hazGAN package, which is a custom implementation
of a Wasserstein GAN with gradient penalty (WGAN-GP) for spatial data. The script uses
the wandb library for logging and tracking experiments. The script also uses the hazGAN
package for data loading, metrics, and plotting.

Requirements:
-------------
    - env: hazGAN
    - GAN configuratino: config-defaults.yaml
    - data: training/18x22/data.nc

Output:
-------
    - best checkpoint: checkpoint.weights.h5
    - saved_models: generator.weights.h5, critic.weights.h5

To use pretrained weights:
--------------------------
>>> import hazGAN
>>> new_gan = hazGAN.WGAN(config)
>>> new_gan.load_weights(os.path.join(rundir, 'checkpoint.weights.h5'))

To run locally:
---------------
>>> micromamba activate hazGAN
>>> python train.py

To run on linux cluster:
------------------------
>>> srun -p Short --pty python train.py --dry-run --cluster
>>> srun -p GPU --gres=gpu:tesla:1 --pty python train.py --dry-run --cluster

To run sweep:
-------------
>>> wandb sweep sweep.yaml
>>> wandb agent alison/hazGAN/<sweep_id>
"""

# %%
import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import wandb
import hazGAN as hg
from hazGAN import WandbMetricsLogger
tf.keras.backend.clear_session()

global rundir
global runname
global force_cpu

# some static variables
data_source = "era5"
res = (18, 22)
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
plot_kwargs = {"bbox_inches": "tight", "dpi": 300}


def check_interactive(sys):
    """Check if running in interactive mode"""
    if hasattr(sys, 'ps1'):
        print("Running interactively")
        return True
    else:
        print("Not running interactively")
        return False
    

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


# %% ----Main function----
def main(config):
    # load data
    data = hg.load_training(datadir,
                            config.train_size,
                            'reflect',
                            gumbel_marginals=config.gumbel,
                            u10_min=config.u10_min
                            )
    
    train_u = data['train_u']
    test_u = data['test_u']
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(config.batch_size)
    print(f"Training data shape: {train_u.shape}")
    
    # define callbacks
    chi_score = hg.ChiScore({
            "train": next(iter(train)),
            "test": next(iter(test))
            },
        frequency=config.chi_frequency,
        gumbel_margins=config.gumbel
        )
    
    chi_squared = hg.ChiSquared(
        batchsize=config.batch_size,
        frequency=config.chi_frequency
        )
    
    compound = hg.CompoundMetric(frequency=config.chi_frequency)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(rundir, "checkpoint.weights.h5"),
        monitor="compound_metric",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
        )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="chi_rmse",
        patience=50,
        mode="min",
        verbose=True,
        restore_best_weights=True,
        start_from_epoch=10
        )

    # compile
    with tf.device(device):
        gan = getattr(hg, f"compile_{config.model}")(config, nchannels=2)
        history = gan.fit(
            train,
            epochs=config.nepochs,
            callbacks=[
                chi_score,
                chi_squared,
                compound,
                WandbMetricsLogger(),
                checkpoint,
                early_stopping
                ]
        )

    final_chi_rmse = history.history['chi_rmse'][-1]
    print(f"Final chi_rmse: {final_chi_rmse}")

    if final_chi_rmse <= 10.0:
        save_config(rundir)

        # ----Figures----
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        train_u = hg.inv_gumbel(hg.unpad(train_u, paddings)).numpy()
        test_u = hg.inv_gumbel(hg.unpad(test_u, paddings)).numpy()
        fake_u = hg.unpad(gan(nsamples=1000), paddings).numpy()
        
        cmap = plt.cm.coolwarm_r
        vmin = 1
        vmax = 2
        cmap.set_under(cmap(0))
        cmap.set_over(cmap(.99))

        # Fig 1: channel extremal coefficients
        def get_channel_ext_coefs(x):
            n, h, w, c = x.shape
            excoefs = hg.get_extremal_coeffs_nd(x, [*range(h * w)])
            excoefs = np.array([*excoefs.values()]).reshape(h, w)
            return excoefs
        
        excoefs_train = get_channel_ext_coefs(train_u)
        excoefs_test = get_channel_ext_coefs(test_u)
        excoefs_gan = get_channel_ext_coefs(fake_u)
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1, 1, 1, .05]}
                            )
        im = ax[0].imshow(excoefs_train, vmin=vmin, vmax=vmax, cmap=cmap)
        im = ax[1].imshow(excoefs_test, vmin=vmin, vmax=vmax, cmap=cmap)
        im = ax[2].imshow(excoefs_gan, vmin=vmin, vmax=vmax, cmap=cmap)
        for a in ax:
            a.set_yticks([])
            a.set_xticks([])
            a.invert_yaxis()
        ax[0].set_title('Train', fontsize=16)
        ax[1].set_title('Test', fontsize=16)
        ax[2].set_title('hazGAN', fontsize=16);
        fig.colorbar(im, cax=ax[3], extend='both', orientation='vertical')
        ax[0].set_ylabel('Extremal coeff', fontsize=18);
        log_image_to_wandb(fig, f"extremal_dependence", imdir)

        # Fig 2: spatial extremal coefficients
        i = 0 # only look at wind speed
        ecs_train = hg.pairwise_extremal_coeffs(train_u.astype(np.float32)[..., i]).numpy()
        ecs_test = hg.pairwise_extremal_coeffs(test_u.astype(np.float32)[..., i]).numpy()
        ecs_gen = hg.pairwise_extremal_coeffs(fake_u.astype(np.float32)[..., i]).numpy()
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.5),
                            gridspec_kw={
                                'wspace': .02,
                                'width_ratios': [1, 1, 1, .05]}
                                )
        
        im = axs[0].imshow(ecs_train, vmin=vmin, vmax=vmax, cmap=cmap)
        im = axs[1].imshow(ecs_test, vmin=vmin, vmax=vmax, cmap=cmap)
        im = axs[2].imshow(ecs_gen, vmin=vmin, vmax=vmax, cmap=cmap)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, cax=axs[3], extend='both', orientation='vertical');
        axs[0].set_title("Train", fontsize=16)
        axs[1].set_title("Test", fontsize=16)
        axs[2].set_title("hazGAN", fontsize=16)
        axs[0].set_ylabel('Extremal coeff.', fontsize=18);
        log_image_to_wandb(fig, f"spatial_dependence", imdir)

        # Fig 3: 64 most extreme samples
        # inverse transform 
        X = data['train_x'].numpy()
        U = hg.unpad(data['train_u']).numpy()
        params = data['params']
        x = hg.POT.inv_probability_integral_transform(fake_u, X, U)

        # plot the 64 samples with highest max winds
        x = x[..., 0]
        maxima = np.max(x, axis=(1, 2))
        idx = np.argsort(maxima)
        x = x[idx, ...]
        lon = np.linspace(80, 95, 22)
        lat = np.linspace(10, 25, 18)
        lon, lat = np.meshgrid(lon, lat)
        fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax in enumerate(axs.ravel()):
            ax.contourf(lon, lat, x[i, ...], cmap='Spectral_r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        fig.suptitle('64 most extreme samples', fontsize=20)

        log_image_to_wandb(fig, f"max_samples", imdir)
    
    else: # delete rundir and its contents
        print("Chi score too high, deleting run directory")
        os.system(f"rm -r {rundir}")

# %% ----Train the model----
if __name__ == "__main__":
    if not check_interactive(sys):
        # parse arguments (if running from command line)
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', '-d', dest="dry_run", action='store_true', default=False, help='Dry run')
        parser.add_argument('--cluster', '-c', dest="cluster", action='store_true', default=False, help='Running on cluster')
        parser.add_argument('--force-cpu', '-f', dest="force_cpu", action='store_true', default=False, help='Force use CPU (for debugging)')
        args = parser.parse_args()
        dry_run = args.dry_run
        cluster = args.cluster
        force_cpu = args.force_cpu
    else:
        # set defaults for interactive mode
        dry_run = True
        cluster = False
        force_cpu = False

    # setup device
    device = config_tf_devices()

    # set up directories
    if cluster:
        wd = os.path.join('/soge-home', 'projects', 'mistral', 'alison', 'hazGAN')
    else:
        wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')  # hazGAN directory

    datadir = os.path.join(wd, 'training', f"{res[0]}x{res[1]}")  # keep data folder in parent directory
    print(f"Loading data from {datadir}")
    imdir = os.path.join(wd, "figures", "temp")

    # initialise wandb
    if dry_run: # doesn't work with sweeps
        print("Starting dry run")
        wandb.init(project="test", mode="disabled")
        wandb.config.update({
            'nepochs': 200,
            'train_size': 96,
            'batch_size': 32,
            'chi_frequency': 1
            },
            allow_val_change=True)
        runname = 'dry-run'
    else:
        wandb.init(allow_val_change=True)  # saves snapshot of code as artifact
        runname = wandb.run.name
    rundir = os.path.join(wd, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)

    # set seed for reproductibility
    wandb.config["seed"] = np.random.randint(0, 1e6)
    tf.keras.utils.set_random_seed(wandb.config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()        # removes stochasticity from individual operations

    main(wandb.config)

# %% 
