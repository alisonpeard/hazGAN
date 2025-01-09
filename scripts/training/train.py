"""
For conditional training (no constant fields yet).
"""
# %% quick settings
DRY_RUN_EPOCHS       = 1
SAMPLES_PER_EPOCH    = 1280     # 1000   # samples per epoch
CONTOUR_PLOT         = False
PROJECT              = 'hazGAN-linux'

# %% actual script
import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
import yaml
import argparse
import random
import wandb
from environs import Env
import numpy as np
import matplotlib.pyplot as plt

import torch
from keras.callbacks import ModelCheckpoint

from hazGAN import plot
from hazGAN.torch import unpad
from hazGAN.torch import load_data # type: ignore
from hazGAN.torch import WGANGP
from hazGAN.torch import MemoryLogger
from hazGAN.torch import WandbMetricsLogger
from hazGAN.torch import ImageLogger
from hazGAN.torch import LRScheduler


global run
global datadir
global rundir
global imdir
global runname
global device
global force_cpu


plot_kwargs = {"bbox_inches": "tight", "dpi": 300}


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
            print("Using MPS")
            return "mps"

        elif torch.cuda.is_available():
            print("Using CUDA")
            return "cuda"

    print("No GPUs available, using CPU")
    return "cpu"


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def summarise_mps_memory():
        current = torch.mps.current_allocated_memory() / 1e9
        driver  = torch.mps.driver_allocated_memory()  / 1e9

        print("\nMemory:\n-------")
        print(f"current allocated: {current:.2f} GB")
        print(f"driver allocated: {driver:.2f} GB\n")
        
        torch.mps.empty_cache()


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
    label_indices = (train.dataset.data['label'] == label).nonzero().reshape(-1)
    train_subset = train.dataset[label_indices.tolist()]
    condition_subset = train_subset['condition'].cpu().numpy()

    # make generated data for same labels and conditions
    nsamples = len(condition_subset)
    condition = condition_subset
    labels = np.tile(label, len(condition))
    lower_bound = np.floor(condition_subset.min())
    upper_bound = np.ceil(condition_subset.max())

    # print specs
    print(
        "\nConditioning on {} {:.2f}mps - {:.2f}mps max wind percentiles with label {}"
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
    try:
        plot.figure_one(fake_u, train_u, valid_u, imdir, id=label)
        plot.figure_two(fake_u, train_u, valid_u, imdir, id=label)
        plot.figure_three(fake_u, train_u, imdir, contour=CONTOUR_PLOT, id=label)
        plot.figure_four(fake_u, train_u, train_x, params, imdir, contour=CONTOUR_PLOT, id=label)
        # plot.figure_five(fake_u, train_u, imdir)                   # augmented    
        export_sample(fake_u)
        
        if len(history) > 1:
            print("\nResults:\n--------")
            final_chi_rmse = history['chi_rmse'][-1]
            print(f"final_chi_rmse: {final_chi_rmse:.4f}")

    except Exception as e:
        print(f"Error generating figures: {e}")


def main(config):
    # load data
    trainloader, validloader, metadata = load_data(datadir, config['batch_size'],
                                       train_size=config['train_size'],
                                       fields=config['fields'],
                                       thresholds=config['thresholds'],
                                       device=device, subset=config['pretrain_size'],)
    
    # update config with number of labels
    config = update_config(config, 'nconditions', len(metadata['labels']))

    # compile model
    model = WGANGP(device=device, **config)
    model.compile()

    # callbacks
    memory_logger = MemoryLogger(100, logdir='logs')
    wandb_logger = WandbMetricsLogger()
    image_logger = ImageLogger()
    checkpointer = ModelCheckpoint(
        os.path.join(rundir, "checkpoint.weights.h5"),
        monitor="chi_rmse",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=True
        )
    
    callbacks = [wandb_logger, image_logger, checkpointer]
    if config['scheduler']:
        scheduler = LRScheduler(
            config['learning_rate'],
            config['epochs'],
            warmup_steps=config['warmup']
            )
        callbacks = [scheduler] + callbacks


    # check memory before starting
    # summarise_mps_memory()

    # fit model
    print("\nTraining...\n")
    history = model.fit(trainloader, epochs=config['epochs'],
                        callbacks=callbacks,
                        steps_per_epoch=(SAMPLES_PER_EPOCH // config['batch_size']),
                        target_weights=torch.tensor(config['target_weights']),
                        validation_data=validloader,
                        weight_update_frequency=config['weight_update_frequency'],
                        )

    # evaluate
    evaluate_results(trainloader, model, 1, config, history.history, metadata)
    evaluate_results(trainloader, model, 2, config, history.history, metadata)
    return history


if __name__ == "__main__":
    # setup environment
    if not check_interactive(sys):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', '-d', dest="dry_run", action='store_true', default=False, help='Dry run')
        parser.add_argument('--force-cpu', '-f', dest="force_cpu", action='store_true', default=False, help='Force use CPU (for debugging)')
        args = parser.parse_args()
        dry_run   = args.dry_run
        force_cpu = args.force_cpu
    else:
        dry_run   = True
        force_cpu = False

    # use GPU if available
    device = config_devices()

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
        run = wandb.init(project=PROJECT, allow_val_change=True, settings=wandb.Settings(_service_wait=300))  # saves snapshot of code as artifact
        runname = wandb.run.name
        config = wandb.config
    
    # format random 
    # config = update_config(config, 'seed', np.random.randint(0, 100))
    # note, sampling won't be fully deterministic on CUDA
    print("Random Seed: ", config['seed'])
    seed_everything(config['seed'])
    
    # make dir to save results
    rundir = os.path.join(workdir, "_wandb-runs", runname)
    os.makedirs(rundir, exist_ok=True)

    # train
    result = main(config)
    print(result)

    # notify("Process finished", "Python script", "Finished making pretraining data")


# %% ---DEBUG BELOW THIS LINE----


# %% ---DEBUG ABOVE THIS LINE----