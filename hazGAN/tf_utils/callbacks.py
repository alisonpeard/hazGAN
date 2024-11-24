import os
from collections import deque
import numpy as np
import wandb
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.nn import sigmoid_cross_entropy_with_logits as cross_entropy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ..extreme_value_theory import chi_loss, inv_gumbel, pairwise_extremal_coeffs, chi2metric
from ..utils import unpad
from ..extreme_value_theory.peak_over_threshold import inv_probability_integral_transform

class WandbMetricsLogger(Callback):
    """
    Custom Wandb callback to log metrics.

    Source: https://community.wandb.ai/t/sweeps-not-showing-val-loss-with-keras/4495"""
    def on_epoch_end(self, epoch, logs=None):
        if wandb.run is not None:
            wandb.log(logs)


class OverfittingDetector(Callback):
    """From arXiv:2006.06676v2 Equation (1a)."""
    def __init__(self, valid, n=4):
        super().__init__()
        self.train = deque([], n)
        self.validation = deque([], n)
        self.generated = deque([], n)
        self.valid = valid

    def on_batch_end(self, batch, logs={}):
        self.train.appendleft(logs.get("critic_real", np.nan))
        self.generated.appendleft(logs.get("critic_fake", np.nan))

        # get critic validation score
        valid_batch = next(iter(self.valid))
        valid_score = tf.reduce_mean(self.model.critic(valid_batch, training=False))
        self.validation.appendleft(valid_score)

        critic_train = np.nanmean(self.train)
        critic_val = np.nanmean(self.validation)
        critic_generated = np.nanmean(self.generated)
        rv = (critic_train - critic_val) / (critic_train - critic_generated)
        logs['rv'] = rv
        if (rv > 0.999) & batch > 8:
            print("\nEarly stopping due to overfitting.\n")
            self.model.stop_training = True

class CountImagesSeen(Callback):
    def __init__(self, batch_size):
        """Add to image counter at end of each batch."""
        super().__init__()
        self.batch_size = batch_size
        self.batches_seen = 0

    def on_batch_end(self, batch, logs={}):
        # get training data
        self.batches_seen += 1
        logs['images_seen'] = int(self.batches_seen * self.batch_size)


class CriticVal(Callback):
    """Monitor critic's performance on the test set."""
    def __init__(self, validation_data, frequency=1):
        super().__init__()
        self.frequency = frequency
        self.validation_data = validation_data
        self.critic_val = None

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            nbatch = 0
            score = 0
            for batch in self.validation_data:
                batch = batch[0] # when using custom batch function
                augmented = self.model.augment(batch)
                score_batch = self.model.critic(augmented, training=False)
                score += tf.reduce_mean(score_batch)
                nbatch += 1
            score = score / nbatch
            logs["critic_val"] = tf.reduce_mean(score).numpy()


class ChiScore(Callback):
    """
    Custom metric for evtGAN to compare extremal coefficients across space.
    """
    def __init__(self, validation_data: dict, frequency=1, gumbel_margins=False):
        super().__init__()
        if gumbel_margins:
            for name, data in validation_data.items():
                validation_data[name] = inv_gumbel(data) # transform to uniform if gumbel margins
        self.validation_data = validation_data
        self.frequency = frequency
        for name, data in validation_data.items():
            for c in range(tf.shape(data)[-1].numpy()):
                data_c = data[..., c]
                extcoeffs = pairwise_extremal_coeffs(data_c)
                setattr(self, f"extcoeffs_{name}_channel_{c}", extcoeffs)
            setattr(self, f"chi_score_{name}", None)


    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            for name, data in self.validation_data.items():
                batch_size = tf.shape(data)[0]
                generated_data = self.model(nsamples=batch_size) # handles inverse gumbel in __call__

                def compute_chi_diff(i):
                    generated_data_i = generated_data[..., i]
                    extcoeff = pairwise_extremal_coeffs(generated_data_i)
                    extcoeff_data = getattr(self, f"extcoeffs_{name}_channel_{i}")
                    return tf.sqrt(tf.reduce_mean(tf.square(extcoeff - extcoeff_data)))

                c = tf.shape(data)[-1]
                chi_diffs = tf.map_fn(compute_chi_diff, tf.range(c), dtype=tf.float32)
                rmse = tf.reduce_mean(chi_diffs).numpy()
                setattr(self, f"chi_score_{name}", rmse)
                logs[f"chi_score_{name}"] = rmse


class ChiSquared(Callback):
    def __init__(self, batchsize, frequency=1, nbins=20):
        super().__init__()
        self.batchsize = batchsize
        self.nbins = nbins
        self.frequency = frequency
        self.chi_squared = None

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0: # run after others
            generated_data = self.model(nsamples=self.batchsize)
            chisq = chi2metric(generated_data, nbins=self.nbins).numpy()

            # updates
            self.chi_squared = chisq
            logs["chi_squared"] = chisq
    

class CompoundMetric(Callback):
    def __init__(self, frequency=1):
        super().__init__()
        self.chi_rmse = None
        self.chi_squared = None
        self.metric = None
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0: # run after others
            chisq = logs.get("chi_squared")
            chirmse = logs.get("chi_score_test")

            if chisq is None or chirmse is None:
                tf.print(
                    "\nWarning: one or more of the metrics is None. Skipping compound metric computation."
                    )
                return
            
            metric = tf.math.log(1 + chisq) + chirmse # compound_metric(chisq, chirmse)
            logs["compound_metric"] = metric.numpy()


@tf.function
def compound_metric(chisq, chirmse, eps=1e-6):
    """Transform to comparable scales"""
    chisq = tf.clip_by_value(chisq, 0, 1 - eps)
    chisq = -tf.math.log(1 - chisq)
    chirmse = tf.math.log(1 + chirmse)
    return chisq + chirmse


class CrossEntropy(Callback):
    def __init__(self, validation_data, wandb_run=None):
        super().__init__()
        self.validation_data = validation_data
        self.d_loss = None
        self.g_loss = None
        self.wandb_run = wandb_run

    def on_epoch_end(self, epoch, logs={}):
        batch_size = tf.shape(self.validation_data)[0]
        generated_data = self.model(batch_size)

        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))

        score_real = self.model.critic(self.validation_data, training=False)
        score_fake = self.model.critic(generated_data, training=False)

        d_loss_real_test = tf.reduce_mean(cross_entropy(labels_real, score_real))
        d_loss_fake_test = tf.reduce_mean(cross_entropy(labels_fake, score_fake))
        d_loss_test = d_loss_real_test + d_loss_fake_test
        g_loss_test = tf.reduce_mean(cross_entropy(labels_real, score_fake))

        # updates
        self.d_loss_test = d_loss_test
        self.g_loss_test = g_loss_test
        logs["d_loss_test"] = d_loss_test
        logs["g_loss_test"] = g_loss_test


class ChannelVisualiser(Callback):
    def __init__(self, frequency=1, channel=0, runname='untitled-run',
                 data: tuple = None):
        super().__init__()
        self.frequency = frequency
        self.generated_images = []
        self.runname = runname
        self.data = data
        self.channel = channel
        print("Warning: resolution hard-coded as 18x22 for ChannelVisualiser callback.")

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.frequency == 0) & (epoch > 0):
            clear_output(wait=True)

            generated_data = unpad(self.model(nsamples=3))
            if self.data is not None:
                X, U = self.data
                X = X.numpy()
                U = unpad(U).numpy()
                generated_data = generated_data.numpy()
                generated_data = inv_probability_integral_transform(generated_data, X, U)
                dist = "original"
            else:
                dist = "Gumbel"
                generated_data = generated_data.numpy()
            
            fig, axs = plt.subplots(1, 3, figsize=(10, 2.5),
                                    gridspec_kw={"wspace": 0},
                                    sharex=True, sharey=True)
            vmin = tf.reduce_min(generated_data)
            vmax = tf.reduce_max(generated_data)
            lats = np.linspace(25, 10, 18)
            lons = np.linspace(80, 95, 22)
            X, Y = np.meshgrid(lons, lats)
            for i, ax in enumerate(axs):
                im = ax.contourf(
                    X,
                    Y,
                    generated_data[i, ..., self.channel],
                    cmap="Spectral_r",
                    levels=20,
                    vmin=vmin,
                    vmax=vmax
                )
                ax.invert_yaxis()
                ax.label_outer()
                ax.set_xlabel('Longitude')
            axs[0].set_ylabel('Latitude')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.suptitle(f"Generated images for {self.runname} for epoch: {epoch} ({dist} scale)")
            log_image_to_wandb(fig, f"channel{self.channel}_{epoch}", dir="imgs")
            # plt.show()


def log_image_to_wandb(fig, name: str, dir: str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath)
    wandb.log({name: wandb.Image(impath)})