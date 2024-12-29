import os
from datetime import datetime
os.environ["KERAS_BACKEND"] = "torch"
import wandb
from torch import mps
from keras.callbacks import Callback


__all__ = ["WandbMetricsLogger", "MemoryLogger"]


class WandbMetricsLogger(Callback):
    """
    Custom Wandb callback to log metrics.

    Source: https://community.wandb.ai/t/sweeps-not-showing-val-loss-with-keras/4495
    """
    def on_epoch_end(self, epoch, logs=None):
        if wandb.run is not None:
            wandb.log(logs)


class MemoryLogger(Callback):
    def __init__(self, frequency, logdir='.', clear_cache=True):
        super().__init__()
        self.clear_cache = clear_cache
        self.path = os.path.join(logdir, "mps.log")
        self.frequency = frequency
        with open(self.path, "w") as stream:
            stream.write("MPS Memory Logger\n------------------\n")
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            current = mps.current_allocated_memory() / 1e9
            driver  = mps.driver_allocated_memory() / 1e9
            logs['mps_current_allocated'] = current
            logs['mps_driver_allocated']  = driver

            with open(self.path, "a") as stream:
                stream.write(f"{datetime.now()} -- ")
                stream.write(f"batch {batch} -- ")
                stream.write(f"current allocated: {current:.2f} GB -- ")
                stream.write(f"driver allocated: {driver:.2f} GB\n")

            mps.synchronize()

            if self.clear_cache:
                mps.empty_cache()
