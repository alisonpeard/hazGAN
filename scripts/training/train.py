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
    - local paths: .env file
    - GAN configuration: config-defaults.yaml
    - data: training/18x22/data.nc

Output:
-------
    - best checkpoint: checkpoint.weights.h5
    - final checkpoint: generator.weights.h5, critic.weights.h5

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
>>> srun -p Short --pty python train.py --dry-run
>>> srun -p GPU --gres=gpu:tesla:1 --pty python train.py --dry-run

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
from environs import Env
import hazGAN as hazzy
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
        print("Running interactively.")
        return True
    else:
        print("Not running interactively.")
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
    minority = hazzy.load_training(datadir,
                            config['train_size'],
                            padding_mode='reflect',
                            gumbel_marginals=config['gumbel'],
                            u10_min=15,
                            channels=config['channels']
                            )
    majority = hazzy.load_training(datadir,
                            config['train_size'],
                            padding_mode='reflect',
                            gumbel_marginals=config['gumbel'],
                            u10_max=15,
                            channels=config['channels']
                            )
    pretrain = hazzy.load_pretraining(datadir,
                                         config['train_size'],
                                         padding_mode='reflect',
                                         gumbel_marginals=config['gumbel'],
                                         channels=config['channels']
                                         )
    train_minority = minority['train_u']
    test_minority = minority['test_u']
    train_majority = majority['train_u'][:100, ...]
    test_majority = majority['test_u'][:100, ...]
    train_pre = pretrain['train_u'][:100, ...]
    test_pre = pretrain['test_u'][:100, ...]
    train = hazzy.BalancedBatchNd([train_pre, train_majority, train_minority],
                               ratios=config['ratios'], infinite=True)
    test = hazzy.BalancedBatchNd([test_pre, test_majority, test_minority],
                               ratios=config['ratios'])
    print("Number of training batches:", len(train))
    print("Number of validation batches:", len(test))

    # define callbacks
    critic_val = hazzy.CriticVal(test)
    image_count = hazzy.CountImagesSeen(ntrain=train.size)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(rundir, "checkpoint.weights.h5"),
        monitor="compound_metric",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
        )
    
    # compile
    print("\nStarting training...")
    with tf.device(device):
        gan = getattr(hazzy,
                      f"compile_{config['model']}")(
                          config,
                          nchannels=len(config['channels']),
                          train=train
                          )
        history = gan.fit(
            train,
            epochs=config.nepochs,
            callbacks=[
                critic_val,
                image_count,
                WandbMetricsLogger(),
                checkpoint
                ]
        )
        gan.cleanup()
    print('Finished training!')
    final_chi_rmse = history.history['chi_rmse'][-1]
    print(f"Final chi_rmse: {final_chi_rmse}")

    if True: #TODO: final_chi_rmse <= 20.0:
        save_config(rundir)
        all_data = minority = hazzy.load_training(datadir,
                            config.train_size,
                            padding_mode='reflect',
                            gumbel_marginals=config['gumbel'],
                            channels=config['channels']
                            )
        train_u = all_data['train_u']
        test_u = all_data['test_u']

        # ----Figures----
        channel = 0
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        train_u = hazzy.inv_gumbel(hazzy.unpad(train_u, paddings)).numpy()
        test_u = hazzy.inv_gumbel(hazzy.unpad(test_u, paddings)).numpy()
        fake_u = hazzy.unpad(gan(nsamples=64), paddings).numpy()
        
        cmap = plt.cm.coolwarm_r
        vmin = 1
        vmax = 2
        cmap.set_under(cmap(0))
        cmap.set_over(cmap(.99))

        # Fig 1: channel extremal coefficients
        def get_channel_ext_coefs(x):
            n, h, w, c = x.shape
            excoefs = hazzy.get_extremal_coeffs_nd(x, [*range(h * w)])
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
        ecs_train = hazzy.pairwise_extremal_coeffs(train_u.astype(np.float32)[..., channel]).numpy()
        ecs_test = hazzy.pairwise_extremal_coeffs(test_u.astype(np.float32)[..., channel]).numpy()
        ecs_gen = hazzy.pairwise_extremal_coeffs(fake_u.astype(np.float32)[..., channel]).numpy()
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

        # ----Fig 3: 64 most extreme generated samples----
        X = all_data['train_x'].numpy()
        U = hazzy.unpad(all_data['train_u']).numpy()
        params = all_data['params']
        x = hazzy.POT.inv_probability_integral_transform(fake_u, X, U, params)

        x = fake_u[:64, ..., channel] # x[..., channel]
        vmin = x.min()
        vmax = x.max()
        if x.shape[0] < 64:
            # repeat x until it has 64 samples
            x = np.concatenate([x] * int(np.ceil(64 / x.shape[0])), axis=0)
        maxima = np.max(x, axis=(1, 2))
        idx = np.argsort(maxima)
        x = x[idx, ...]
        lon = np.linspace(80, 95, 22)
        lat = np.linspace(10, 25, 18)
        lon, lat = np.meshgrid(lon, lat)
        fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax in enumerate(axs.ravel()):
            im = ax.contourf(lon, lat, x[i, ...], cmap='Spectral_r',
                        vmin=vmin, vmax=vmax, levels=20)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        fig.suptitle(f'64 most extreme {config.channels[channel]} generated samples', fontsize=18)
        fig.colorbar(im, ax=axs.ravel().tolist())
        log_image_to_wandb(fig, f"max_samples", imdir)

        # ----Fig 4: 64 most extreme training samples----
        # X = all_data['train_x'].numpy()[..., channel]
        X = hazzy.unpad(all_data['train_u']).numpy()[:64, ..., channel]
        X = hazzy.inv_gumbel(X) # fix cbar 23-11-2024
        vmin = X.min()
        vmax = X.max()
        if X.shape[0] < 64:
            # repeat X until it has 64 samples
            X = np.concatenate([X] * int(np.ceil(64 / X.shape[0])), axis=0)
        maxima = np.max(X, axis=(1, 2))
        idx = np.argsort(maxima)
        X = X[idx, ...]
        fig, axs = plt.subplots(8, 8, figsize=(10, 8), sharex=True, sharey=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, ax in enumerate(axs.ravel()):
            im = ax.contourf(lon, lat, X[i, ...], cmap='Spectral_r',
                        vmin=vmin, vmax=vmax, levels=20)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        fig.colorbar(im, ax=axs.ravel().tolist())
        fig.suptitle(f'64 most extreme {config.channels[channel]} training samples', fontsize=18)
        log_image_to_wandb(fig, f"max_train_samples", imdir)
    
        # ---Save a sample for further testing---
        fake_u = hazzy.unpad(gan(nsamples=1000), paddings).numpy()
        np.savez(os.path.join(rundir, 'samples.npz'), uniform=fake_u)

    else: # delete rundir and its contents
        print("Chi score too high, deleting run directory")
        os.system(f"rm -r {rundir}")
    
    return history.history

# %% ----Train the model----
if __name__ == "__main__":
    if not check_interactive(sys):
        # parse arguments (if running from command line)
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', '-d', dest="dry_run", action='store_true', default=False, help='Dry run')
        parser.add_argument('--force-cpu', '-f', dest="force_cpu", action='store_true', default=False, help='Force use CPU (for debugging)')
        args = parser.parse_args()
        dry_run = args.dry_run
        force_cpu = args.force_cpu
    else:
        # set defaults for interactive mode
        dry_run = True
        force_cpu = False

    # setup device
    device = config_tf_devices()

    # set up directories
    env = Env()
    env.read_env(recurse=True)  # read .env file, if it exists
    wd = env.str("WORKINGDIR")
    datadir = os.path.join(wd, 'training', f"{res[0]}x{res[1]}")  # keep data folder in parent directory
    print(f"Loading data from {datadir}")
    imdir = os.path.join(wd, "figures", "temp")

    # initialise wandb
    if dry_run: # doesn't work with sweeps
        print("Starting dry run")
        wandb.init(project="test", mode="disabled")
        wandb.config.update({
            'nepochs': 10,
            'train_size': 0.6,
            'batch_size': 64,
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
    wandb.config["seed"] = np.random.randint(0, 100)
    tf.keras.utils.set_random_seed(wandb.config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()        # removes stochasticity from individual operations
    config = wandb.config
    # %%
    history = main(config)

# %% ---------------------------------END---------------------------------

