"""
For conditional training (no constants yet).
"""
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
from hazGAN import plots
from hazGAN import WandbMetricsLogger

plot_kwargs = {"bbox_inches": "tight", "dpi": 300}

tf.keras.backend.clear_session()
# ? add memory growth ?

RUN_EAGERLY = False
MIN_CHI_RMSE = 1000.

# globals
global datadir
global rundir
global imdir
global runname
global force_cpu


def check_interactive(sys):
    """Useful check for VS code."""
    if hasattr(sys, 'ps1'):
        print("\nRunning interactive session.\n")
        return True
    else:
        print("\nNot running interactive session.\n")
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


def update_config(config, key, value):
    """Wrapper for updating config values."""
    if isinstance(config, wandb.sdk.wandb_config.Config):
        config.update({key: value}, allow_val_change=True)
    elif isinstance(config, dict):
        config[key] = value
    else:
        raise ValueError("config must be either a dict or a wandb Config object.")
    return config


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


def export_sample(samples):
    np.savez(os.path.join(rundir, 'samples.npz'), uniform=samples)


def evaluate_results(train,
                     valid,
                     config,
                     history,
                     model,
                     metadata
                     ) -> None:
    """Make some key figures to view results."""
    #! This should work ok but only generated samples are filtered by label
    final_chi_rmse = history['chi_rmse'][-1]
    print(f"\nFinished training!\n")
    if final_chi_rmse <= MIN_CHI_RMSE:
        save_config(rundir, config)
        print("Gathering labels and conditions...")
        biggest_label = metadata['labels'][-1]
        train_extreme = train.take(1000).unbatch().filter(lambda sample: sample['label']==biggest_label)
        # valid_extreme = valid.unbatch().filter(lambda sample: sample['label']==biggest_label)
        condition = np.array(list(x['condition'] for x in train_extreme.as_numpy_iterator()))
        
        #! very crude condition interpolation to get 1000 realistic conditions
        x = np.linspace(0, 100, 1000)
        xp = np.linspace(0, 100, len(condition))
        fp = sorted(condition)
        condition = np.interp(x, xp, fp)
        label = np.tile(biggest_label, 1000)
        print("\nConditioning on 1000 {:.2f} - {:.2f} max wind percentiles".format(condition.min(), condition.max()))
        print("Conditioning on label: {}".format(label[0]))

        print("\nGenerating samples...")
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]) # different for arrays and datasets
        fake_u = hazzy.unpad(model(condition, label, nsamples=1000), paddings=paddings).numpy()

        #TODO: inverse transform train_u to make sure that's working
        # train_u = np.array(list(x['uniform'] for x in train_extreme.as_numpy_iterator()))
        # valid_u = np.array(list(x['uniform'] for x in valid_extreme.as_numpy_iterator()))
        # train_u = hazzy.unpad(train_u, paddings=paddings).numpy()
        # valid_u = hazzy.unpad(valid_u, paddings=paddings).numpy()

        print("\nGathering validation data...")
        train_u = metadata['train']['uniform'].data
        train_x = metadata['train']['anomaly'].data
        valid_u = metadata['valid']['uniform'].data
        params = metadata['train']['params'].data
        
        print("\nValidation data shapes:\n----------------------")
        print("train_u.shape: {}".format(train_u.shape))
        print("fake_u.shape: {}".format(fake_u.shape))
        print("params.shape: {}".format(params.shape))

        print("\nGenerating figures...")
        plots.figure_one(fake_u, train_u, valid_u, imdir)
        plots.figure_two(fake_u, train_u, valid_u, imdir)
        plots.figure_three(fake_u, train_u, imdir)                  # gumbel
        plots.figure_four(fake_u, train_u, train_x, params, imdir)  # full-scale
        plots.figure_five(fake_u, train_u, imdir)                   # augmented    
        export_sample(fake_u)
    else:
        print("Chi score too high, deleting run directory")
        os.system(f"rm -r {rundir}")
    
    print("\nResults:\n--------")
    print(f"final_chi_rmse: {final_chi_rmse:.4f}")


def main(config, verbose=True):
    # load data
    train, valid, metadata = hazzy.load_data(
        datadir,
        label_ratios=config['label_ratios'], 
        batch_size=config['batch_size'],
        train_size=config['train_size'],
        channels=config['channels'],
        gumbel=config['gumbel']
        )
    
    config = update_config(config, 'nconditions', len(metadata['labels']))

    # number of epochs calculations (1 step == 1 batch)
    steps_per_epoch = 5 if dry_run else 200_000 // config['batch_size'] # good rule of thumb
    total_steps = config['epochs'] * steps_per_epoch
    images_per_epoch = steps_per_epoch * config['batch_size']
    total_images = total_steps * config['batch_size']

    if verbose:
        print("Training summary:\n-----------------")
        print("Batch size: {:,.0f}".format(config['batch_size']))
        print("Steps per epoch: {:,.0f}".format(steps_per_epoch))
        print("Total steps: {:,.0f}".format(total_steps))
        print("Images per epoch: {:,.0f}".format(images_per_epoch))
        print("Total number of epochs: {:,.0f}".format(config['epochs']))
        print("Total number of training images: {:,.0f}\n".format(total_images))

    # callbacks
    image_count = hazzy.CountImagesSeen(config['batch_size'])
    wandb_logger = WandbMetricsLogger()

    # train
    tf.config.run_functions_eagerly(RUN_EAGERLY) #for debugging
    cgan = hazzy.conditional.compile_wgan(config, nchannels=len(config['channels']))
    history = cgan.fit(train, epochs=config['epochs'],
                       steps_per_epoch=steps_per_epoch,
                       validation_data=valid,
                       callbacks=[image_count, wandb_logger])
    
    evaluate_results(train, valid, config, history.history, cgan, metadata)

    return history.history


if __name__ == "__main__":
    # setup environment
    if not check_interactive(sys):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', '-d', dest="dry_run", action='store_true', default=False, help='Dry run')
        parser.add_argument('--force-cpu', '-f', dest="force_cpu", action='store_true', default=False, help='Force use CPU (for debugging)')
        args = parser.parse_args()
        dry_run = args.dry_run
        force_cpu = args.force_cpu
    else:
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

    # intialise configuration
    if dry_run:
        with open(os.path.join(os.path.dirname(__file__), "config-defaults.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)
        config = {key: value['value'] for key, value in config.items()}
        config = update_config(config, 'epochs', 2)
        runname = "dry-run"
    else:
        wandb.init(allow_val_change=True)  # saves snapshot of code as artifact
        runname = wandb.run.name
        config = wandb.config
    
    config = update_config(config, 'seed', np.random.randint(0, 100))
    config = update_config(
        config,
        'label_ratios',
        {float(k) if k.isnumeric() else k: v for k, v in config['label_ratios'].items()}
        )
    rundir = os.path.join(workdir, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)
    print("\nSampling ratios: {}".format(config['label_ratios']))
    
    # make reproducible
    tf.keras.utils.set_random_seed(config["seed"])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()  # removes stochasticity from individual operations

    # train
    history = main(config)

    # system notify 
    def notify(title, subtitle, message):
        os.system("""
                    osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
                    """.format(message, title, subtitle))
    try:
        notify("Process finished", "Python script", "Finished making pretraining data")
    except Exception as e:
        print("Notification failed: {}".format(e))

# %% dev
