"""
For conditional training (no constant fields yet).
"""
# %% quick settings
DRY_RUN_EPOCHS       = 20
EVAL_CHANNEL         = 2
SUBSET_SIZE          = 200
CONTOUR_PLOT         = False

# %% actual script
import os
import sys
import yaml
import time
import argparse
import wandb
from environs import Env
import numpy as np
import torch

import hazGAN
from hazGAN import plot
from hazGAN.torch import (
    unpad,
    WGANGP,
    load_data,
    MemoryLogger,
    WandbMetricsLogger
)

plot_kwargs = {"bbox_inches": "tight", "dpi": 300}

global run
global datadir
global rundir
global imdir
global runname
global device
global device_name
global force_cpu


def notify(title, subtitle, message):
    os.system("""
              osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
              """.format(message, title, subtitle))


def check_interactive(sys):
    """Useful check for VS code."""
    if hasattr(sys, 'ps1'):
        print("\nRunning interactive session.\n")
        return True
    else:
        print("\nNot running interactive session.\n")
        return False


def config_devices():
    """Use GPU if available and set memory configuration."""
    if not force_cpu:
        if torch.mps.is_available():
            print("Using MPS for memory management")
            device = "mps"

        elif torch.cuda.is_available():
            print("Using CUDA")
            device = "cuda"

        else:
            print("No MPS available, using CPU")
            device = "cpu"

        return getattr(torch, device), device


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


def evaluate_results(train, model, label:int, config:dict,
                     history:dict, metadata:dict, nsamples:int=100
                     ) -> None:
    """Make some key figures to view results.
    """
    save_config(rundir, config)
    print("Gathering labels and conditions...")

    # filter training data
    label_indices = (train.dataset.data['label'] == label).nonzero()[0]
    train_subset = train.dataset[label_indices]
    condition_subset = train_subset['condition'].cpu().numpy()

    # filter generated data
    x  = np.linspace(0, 100, nsamples)
    xp = np.linspace(0, 100, len(condition_subset))
    fp = sorted(condition_subset)
    condition = np.interp(x, xp, fp).astype(np.float32) # interpolate conditions
    condition = condition.reshape(-1, 1)
    labels = np.tile(label, nsamples)
    lower_bound = np.floor(condition_subset.min())
    upper_bound = np.ceil(condition_subset.max())

    # print specs
    print(
        "\nConditioning on {} {:.2f} - {:.2f} max wind percentiles with label {}"
        .format(
            nsamples,
            lower_bound,
            upper_bound,
            label
        )
    )

    print("\nGenerating samples...")
    fake_u = model(label=labels, condition=condition, nsamples=nsamples).detach().cpu().numpy()
    fake_u = unpad(fake_u) # now permute
    fake_u = np.transpose(fake_u, (0, 2, 3, 1))
    print("fake_u.shape: {}".format(fake_u.shape))

    # get metadata to compate
    print("\nGathering validation data...")
    train_u = metadata['train']['uniform'].data
    train_x = metadata['train']['anomaly'].data
    valid_u = metadata['valid']['uniform'].data
    valid_x = metadata['valid']['anomaly'].data
    params = metadata['train']['params'].data

    print("\nFiltering by lower bound on anomaly...")
    train_mask = train_x[..., 0].max(axis=(1, 2)) > lower_bound
    valid_mask = valid_x[..., 0].max(axis=(1, 2)) > lower_bound
    train_u = train_u[train_mask]
    train_x = train_x[train_mask]
    valid_u = valid_u[valid_mask]

    print("\nValidation data shapes:\n----------------------")
    print("train_u.shape: {}".format(train_u.shape))
    print("fake_u.shape: {}".format(fake_u.shape))
    print("params.shape: {}".format(params.shape))

    print("\nGenerating figures...")
    plot.figure_one(fake_u, train_u, valid_u, imdir)
    plot.figure_two(fake_u, train_u, valid_u, imdir)
    plot.figure_three(fake_u, train_u, imdir, contour=CONTOUR_PLOT)
    plot.figure_four(fake_u, train_u, train_x, params, imdir, contour=CONTOUR_PLOT)
    # plot.figure_five(fake_u, train_u, imdir)                   # augmented    
    export_sample(fake_u)
    
    if len(history) > 1:
        print("\nResults:\n--------")
        final_chi_rmse = history['chi_rmse'][-1]
        print(f"final_chi_rmse: {final_chi_rmse:.4f}")



def main(config, verbose=True):
    # load data
    train, valid, metadata = load_data(datadir, config['batch_size'],
                                       train_size=config['train_size'],
                                       fields=config['fields'],
                                       label_ratios=config['label_ratios'],
                                       device=device_name, subset=SUBSET_SIZE)
    
    # update config with number of labels
    config = update_config(config, 'nconditions', len(metadata['labels']))

    # callbacks
    memory_logger = MemoryLogger(100, logdir='logs')
    wandb_logger = WandbMetricsLogger()
    # image_logger = ImageLogger()

    # compile model
    model = WGANGP(config)
    model.compile()

    # fit model
    start = time.time()
    print("\nTraining...\n")
    history = model.fit(train, epochs=config['epochs'], callbacks=[memory_logger, wandb_logger])
    # history = model.train_step(next(iter(train))) # single step
    print("\nFinished! Training time: {:.2f} seconds\n".format(time.time() - start))

    evaluate_results(train, model, EVAL_CHANNEL, config, history.history, metadata)

    return history


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
    device, device_name = config_devices()

    # define paths
    env     = Env()
    env.read_env(recurse=True)
    workdir = env.str("WORKINGDIR")
    datadir = env.str('TRAINDIR')
    imdir   = os.path.join(workdir, "figures", "temp")

    # intialise config object
    if dry_run:
        with open(os.path.join(os.path.dirname(__file__), "config-defaults.yaml"), 'r') as stream:
            config = yaml.safe_load(stream)
        config = {key: value['value'] for key, value in config.items()}
        run = None
        config = update_config(config, 'epochs', DRY_RUN_EPOCHS)
        runname = "dry-run"
    else:
        run = wandb.init(allow_val_change=True, settings=wandb.Settings(_service_wait=300))  # saves snapshot of code as artifact
        runname = wandb.run.name
        config = wandb.config
    
    # format config
    config = update_config(config, 'seed', np.random.randint(0, 100))
    config = update_config(
        config,
        'label_ratios',
        {float(k) if k.isnumeric() else k: v for k, v in config['label_ratios'].items()}
        )
    print("\nSampling ratios: {}".format(config['label_ratios']))
    
    # make dir to save results
    rundir = os.path.join(workdir, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)

    # train
    device.empty_cache()
    result = main(config)

    notify("Process finished", "Python script", "Finished making pretraining data")


# %% ---DEBUG BELOW THIS LINE----

from hazGAN.constants import SAMPLE_CONFIG

model = WGANGP(SAMPLE_CONFIG)
x = model.call(label=1, nsamples=1, condition=1.)
x_unpadddd = unpad(x)
print(x_unpadddd.shape)
# %% ---DEBUG ABOVE THIS LINE----