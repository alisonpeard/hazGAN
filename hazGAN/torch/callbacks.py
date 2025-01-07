import os
os.environ["KERAS_BACKEND"] = "torch"
from warnings import warn 
from datetime import datetime
import wandb
import numpy as np
from torch import mps
from keras import ops, callbacks
from keras.src import backend
from keras.src.utils import io_utils
from keras.optimizers.schedules import CosineDecay
from keras.callbacks import Callback
from IPython.display import clear_output

from .utils import unpad


__all__ = ["WandbMetricsLogger", "MemoryLogger", "ImageLogger", "LRScheduler"]


class WandbMetricsLogger(Callback):
    """
    Custom Wandb callback to log metrics.

    Source: https://community.wandb.ai/t/sweeps-not-showing-val-loss-with-keras/4495
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if wandb.run is not None:
            wandb.log(logs)


class LRScheduler(callbacks.LearningRateScheduler):
    def __init__(self, lr:float, epochs:int, samples:int=1, warmup_steps=0.1,
                 alpha=1e-6, initial_lr=1e-6, verbose=1):

        total_steps = epochs * samples

        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps * total_steps)

        if total_steps > warmup_steps:
            decay_steps = total_steps - warmup_steps
            cosine_scheduler = CosineDecay(
                initial_lr,
                decay_steps,
                alpha=alpha,
                name="CosineDecay",
                warmup_target=lr,
                warmup_steps=warmup_steps
            )
        else:
            warn("Total steps are less than warmup steps. Skipping warmup.")
            decay_steps = total_steps
            cosine_scheduler = CosineDecay(
                lr,
                decay_steps,
                alpha=alpha,
                name="CosineDecay"
            )

        def float_scheduler(epoch):
            """Requires scheduler returns a float."""
            return float(cosine_scheduler(epoch))
        
        super().__init__(float_scheduler, verbose=verbose)


    def on_epoch_begin(self, epoch, logs=None):
        for optimizer in [self.model.generator_optimizer, self.model.critic_optimizer]:
            try:  # new API
                learning_rate = float(
                    backend.convert_to_numpy(optimizer.learning_rate)
                )
                learning_rate = self.schedule(epoch, learning_rate)
            except TypeError:  # Support for old API for backward compatibility
                learning_rate = self.schedule(epoch)

            if not isinstance(learning_rate, (float, np.float32, np.float64)):
                raise ValueError(
                    "The output of the `schedule` function should be a float. "
                    f"Got: {learning_rate}"
                )

            optimizer.learning_rate = learning_rate
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {learning_rate}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate_critic"] = float(
            backend.convert_to_numpy(self.model.critic_optimizer.learning_rate)
        )
        logs["learning_rate_generator"] = float(
            backend.convert_to_numpy(self.model.generator_optimizer.learning_rate)
        )



class MemoryLogger(Callback):
    def __init__(self, frequency, logdir='.', clear_cache=True):
        super().__init__()
        self.clear_cache = clear_cache
        self.path = os.path.join(logdir, "mps.log")
        self.frequency = frequency
        with open(self.path, "w") as stream:
            stream.write("MPS Memory Logger\n------------------\n")

    def on_batch_begin(self, batch, logs=None):
        if batch % self.frequency == 0:
            current = mps.current_allocated_memory() / 1e9
            driver  = mps.driver_allocated_memory() / 1e9
            logs['mps_current_allocated'] = current
            logs['mps_driver_allocated']  = driver

            with open(self.path, "a") as stream:
                stream.write(f"{datetime.now()} -- ")
                stream.write(f"batch {batch} end -- ")
                stream.write(f"current allocated: {current:.2f} GB -- ")
                stream.write(f"driver allocated: {driver:.2f} GB\n")

            mps.synchronize()

            if self.clear_cache:
                mps.empty_cache()


class ImageLogger(Callback):
    """Log images every n epochs"""
    def __init__(self, frequency:int=1, field:int=0, nsamples:int=8,
                 conditions:np.array=None, labels:list=None, noise:np.array=None
                 ) -> None:
        super().__init__()
        self.frequency = frequency
        self.field = field

        if conditions is None:
            conditions = np.linspace(20, 60, nsamples)
        
        if labels is None:
            labels = np.array([2] * nsamples)
   
        self.nsamples = nsamples
        self.conditions = conditions
        self.labels = labels
        self.seed = 42
        self.noise = noise

    def _sample(self) -> np.array:
        generated = self.model(label=self.labels, condition=self.conditions, noise=self.noise)
        generated = generated.detach().cpu().numpy()
        generated = unpad(generated)
        generated = generated[:, self.field, ::-1, :]
        return generated

    def on_train_begin(self, logs:dict={}) -> None:
        """Initialise fixed noise."""
        if self.noise is None:
            self.noise = self.model.latent_space_distn(
                (self.nsamples, self.model.latent_dim),
                seed=self.seed
                )
    
    def on_epoch_end(self, epoch:int, logs:dict={}) -> None:
        if (epoch % self.frequency == 0):
            clear_output(wait=True)
            generated = self._sample()
            generated = generated * 127.5 + 127.5
            images = np.clip(generated, 0, 255)
            images = images.astype(np.int64)
            wandb_images = [wandb.Image(img) for img in images]

            if wandb.run is not None:
                wandb.log({
                "generated_images": wandb_images,
                "epoch": epoch
                })


