import os
os.environ["KERAS_BACKEND"] = "torch"
import wandb
from keras.callbacks import Callback


class WandbMetricsLogger(Callback):
    """
    Custom Wandb callback to log metrics.

    Source: https://community.wandb.ai/t/sweeps-not-showing-val-loss-with-keras/4495
    """
    def on_epoch_end(self, epoch, logs=None):
        if wandb.run is not None:
            wandb.log(logs)