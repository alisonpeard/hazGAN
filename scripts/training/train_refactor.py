"""For conditional training (no constants yet)."""
# %%
import os
import sys
import yaml
import argparse
import wandb
from environs import Env
import numpy as np
import tensorflow as tf
import hazGAN as hazzy
from hazGAN import WandbMetricsLogger
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()
plot_kwargs = {"bbox_inches": "tight", "dpi": 300}

# declare globals
global datadir
global rundir
global imdir
global runname
global force_cpu


def check_interactive(sys):
    """Useful check for VS code."""
    if hasattr(sys, 'ps1'):
        print("\nRunning interactive session.")
        return True
    else:
        print("Not running interactive session.")
        return False


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


def save_config(dir, config):
    """Wrapper for saving training configuration."""

    configfile = open(os.path.join(dir, "config-defaults.yaml"), "w")

    if not isinstance(config, dict):
        config = config.as_dict()
    
    configdict = {
        key: {"value": value} for key, value in config.items()
    }
    yaml.dump(configdict, configfile)
    configfile.close()


def log_image_to_wandb(fig, name: str, dir: str):
    if wandb.run is not None:
        impath = os.path.join(dir, f"{name}.png")
        fig.savefig(impath, **plot_kwargs)
        wandb.log({name: wandb.Image(impath)})
    else:
        print("Not logging figure, wandb not intialised.")


def figure_one(fake_u:np.array, train_u:np.array, valid_u:np.array):
    """Plot cross-channel extremal coefficients."""
    def get_channel_ext_coefs(x):
            _, h, w, _ = x.shape
            excoefs = hazzy.get_extremal_coeffs_nd(x, [*range(h * w)])
            excoefs = np.array([*excoefs.values()]).reshape(h, w)
            return excoefs
    
    excoefs_train = get_channel_ext_coefs(train_u)
    excoefs_valid = get_channel_ext_coefs(valid_u)
    excoefs_fake = get_channel_ext_coefs(fake_u)

    cmap = plt.cm.coolwarm_r
    vmin = 1
    vmax = 2
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1, 1, 1, .05]}
                            )
    im = ax[0].imshow(excoefs_train, vmin=vmin, vmax=vmax, cmap=cmap)
    im = ax[1].imshow(excoefs_valid, vmin=vmin, vmax=vmax, cmap=cmap)
    im = ax[2].imshow(excoefs_fake, vmin=vmin, vmax=vmax, cmap=cmap)

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


def figure_two(fake_u:np.array, train_u:np.array, valid_u:np.array, channel=0):
    """Plot spatial extremal coefficients."""
    ecs_train = hazzy.pairwise_extremal_coeffs(train_u.astype(np.float32)[..., channel]).numpy()
    ecs_valid = hazzy.pairwise_extremal_coeffs(valid_u.astype(np.float32)[..., channel]).numpy()
    ecs_fake = hazzy.pairwise_extremal_coeffs(fake_u.astype(np.float32)[..., channel]).numpy()
    
    cmap = plt.cm.coolwarm_r
    vmin = 1
    vmax = 2
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))
    
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1, 1, 1, .05]}
                            )

    im = axs[0].imshow(ecs_train, vmin=vmin, vmax=vmax, cmap=cmap)
    im = axs[1].imshow(ecs_valid, vmin=vmin, vmax=vmax, cmap=cmap)
    im = axs[2].imshow(ecs_fake, vmin=vmin, vmax=vmax, cmap=cmap)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im, cax=axs[3], extend='both', orientation='vertical');
    axs[0].set_title("Train", fontsize=16)
    axs[1].set_title("Test", fontsize=16)
    axs[2].set_title("hazGAN", fontsize=16)
    axs[0].set_ylabel('Extremal coeff.', fontsize=18);

    log_image_to_wandb(fig, f"spatial_dependence", imdir)


def figure_three(fake_u, train_u, channel=0,
                 cmap="Spectral_r", levels=20):
    """Plot the 32 most extreme train and generated percentiles."""
    # prep data to plot
    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    fake = fake_u[..., channel]
    real = train_u[..., channel]

    if fake.shape[0] < 32:
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )

    fake_maxima = np.max(fake, axis=(1, 2))
    fake_sorting = np.argsort(fake_maxima)
    fake = fake[fake_sorting, ...]

    real_maxima = np.max(real, axis=(1, 2))
    real_sorting = np.argsort(real_maxima)
    real = real[real_sorting, ...]

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plot specs
    fig = plt.figure(figsize=(16, 16), layout="tight")
    subfigs = fig.subfigures(2, 1, hspace=0.2)

    for subfig, item in zip(subfigs, samples.items()):
        axs = subfig.subplots(4, 8, sharex=True, sharey=True,
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
        label = item[0]
        sample = item[1]
        vmin = sample.min()
        vmax = sample.max()
        for i, ax in enumerate(axs.flat):
            im = ax.contourf(lon, lat, sample[i, ...],
                             vmin=vmin, vmax=vmax,
                             cmap=cmap, levels=levels)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Percentiles')
    log_image_to_wandb(fig, f"max_train_samples", imdir)


def figure_four(fake_u, train_u, train_x, params,
                channel=0, cmap="Spectral_r", levels=20):
    """Plot the 32 most extreme train and generated percentiles."""
    # prep data to plot
    fake = hazzy.POT.inv_probability_integral_transform(fake_u, train_u, train_x, params)
    fake = fake[..., channel]
    real = train_x[..., channel]

    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    fake = fake_u[..., channel]
    real = train_u[..., channel]

    if fake.shape[0] < 32:
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )

    fake_maxima = np.max(fake, axis=(1, 2))
    fake_sorting = np.argsort(fake_maxima)
    fake = fake[fake_sorting, ...]

    real_maxima = np.max(real, axis=(1, 2))
    real_sorting = np.argsort(real_maxima)
    real = real[real_sorting, ...]

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plot specs
    fig = plt.figure(figsize=(16, 16), layout="tight")
    subfigs = fig.subfigures(2, 1, hspace=0.2)

    for subfig, item in zip(subfigs, samples.items()):
        axs = subfig.subplots(4, 8, sharex=True, sharey=True,
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
        label = item[0]
        sample = item[1]
        vmin = sample.min()
        vmax = sample.max()
        for i, ax in enumerate(axs.flat):
            im = ax.contourf(lon, lat, sample[i, ...],
                             vmin=vmin, vmax=vmax,
                             cmap=cmap, levels=levels)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Percentiles')
    log_image_to_wandb(fig, f"max_train_samples", imdir)


def export_sample(samples):
    np.savez(os.path.join(rundir, 'samples.npz'), uniform=samples)


def evaluate_results(train,
                     config,
                     history,
                     model,
                     metadata
                     ) -> None:
    """Make some key figures to view results."""
    #! This should work but only generated samples filtered rn
    final_chi_rmse = history['chi_rmse'][-1]
    print(f"Finished training! chi_rmse: {final_chi_rmse}")
    if final_chi_rmse <= 20.0:
        save_config(rundir, config)
        paddings = metadata['paddings']

        # look at biggest label for everything
        biggest_label = metadata['labels'][-1]
        train_extreme = train.unbatch().filter(lambda sample: sample['label']==biggest_label) #!
        condition = np.array(list(x['condition'] for x in train_extreme.as_numpy_iterator())) # bit slow
        
        #! very crude condition interpolation - do better later
        xp = np.linspace(0, 100, len(condition))
        fp = condition
        x = np.linspace(0, 100, 1000)
        condition = np.interp(x, xp, fp)
        label = np.tile(biggest_label, 1000)

        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]) # different for arrays and datasets
        fake_u = hazzy.unpad(model(condition, label, nsamples=1000), paddings=paddings).numpy()
        train_u = metadata['train']['uniform'].data
        train_x = metadata['train']['anomaly'].data
        valid_u = metadata['valid']['uniform'].data
        params = metadata['train']['params'].data

        print("train_u.shape: {}".format(train_u.shape))
        print("fake_u.shape: {}".format(fake_u.shape))
        print("params.shape: {}".format(params.shape))

        figure_one(fake_u, train_u, valid_u)
        figure_two(fake_u, train_u, valid_u)
        figure_three(fake_u, train_u)                  # gumbel
        figure_four(fake_u, train_u, train_x, params)  # full-scale
        export_sample(fake_u)
    else:
        print("Chi score too high, deleting run directory")
        os.system(f"rm -r {rundir}")


def main(config, verbose=True):
    # load data
    train, valid, metadata = hazzy.load_data(
        datadir,
        label_ratios={'pre':1/3, 7: 1/3, 20: 1/3}, #TODO: config['label_ratios']
        batch_size=64                             #TODO: condig['batch_size']
        ) #TODO: other variants
    config['nconditions'] = len(metadata['labels'])

    # number of epochs calculations
    steps_per_epoch = 20 #TODO: config['steps_per_epoch']
    batch_size = train._input_dataset._batch_size.numpy()
    number_train_images = 10_000 # 300_000 #TODO: config['number_train_imahes']
    number_train_batches = number_train_images // batch_size
    epochs = number_train_batches // steps_per_epoch

    if verbose:
        print("Batch size: {:,.0f}".format(batch_size))
        print("Steps per epoch: {:,.0f}".format(steps_per_epoch))
        print("Training {:,.0f} images".format(number_train_images))
        print("Total number of batches: {:,.0f}".format(number_train_batches))
        print("Training for {:,.0f} epochs".format(epochs))

    # callbacks
    image_count = hazzy.CountImagesSeen(batch_size)
    wandb_logger = WandbMetricsLogger()

    # train
    cgan = hazzy.conditional.compile_wgan(config) # TODO: getattr(hazzy, config['type']).compile_wgan
    history = cgan.fit(train, epochs=1, steps_per_epoch=1,
                       validation_data=valid,
                       callbacks=[image_count, wandb_logger])
    
    evaluate_results(train, config, history.history, cgan, metadata)
    return history.history


if __name__ == "__main__":
    # setup environment
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

    # use GPU if available
    device = config_tf_devices()

    # define paths
    env = Env()
    env.read_env(recurse=True)
    workdir = env.str("WORKINGDIR")
    datadir = env.str('TRAINDIR')
    imdir = os.path.join(workdir, "figures", "temp")

    # intialise hyperparameters
    if dry_run:
        with open(os.path.join(os.path.dirname(__file__), "config-defaults.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)
        config = {key: value['value'] for key, value in config.items()}
        runname = "dry-run"
        config['seed'] = np.random.randint(0, 100)
    else:
        wandb.init(allow_val_change=True)  # saves snapshot of code as artifact
        runname = wandb.run.name
        wandb.config["seed"] = np.random.randint(0, 100)
        config = wandb.config
    rundir = os.path.join(workdir, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)
    
    # make reproducible
    tf.keras.utils.set_random_seed(config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()  # removes stochasticity from individual operations

    # train
    history = main(config)

# %% dev