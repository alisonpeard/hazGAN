import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.nn import sigmoid_cross_entropy_with_logits as cross_entropy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .extreme_value_theory import chi_loss, inv_gumbel, pairwise_extremal_coeffs


class Visualiser(Callback):
    def __init__(self, frequency=1, runname='untitled'):
        super().__init__()
        self.frequency = frequency
        self.generated_images = []
        self.runname = runname

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.frequency == 0) & (epoch > 0):
            clear_output(wait=True)
            nchan = tf.shape(self.model(nsamples=3))[-1].numpy()
            fig, axs = plt.subplots(nchan, 3, figsize=(10, 2 * nchan))
            if nchan == 1:
                axs = axs[tf.newaxis, :]
            generated_data = self.model(nsamples=3)
            vmin = tf.reduce_min(generated_data)
            vmax = tf.reduce_max(generated_data)
            for c in range(nchan):
                for i, ax in enumerate(axs[c, :]):
                    im = ax.imshow(
                        generated_data[i, ..., c].numpy(),
                        cmap="Spectral_r",
                        vmin=vmin,
                        vmax=vmax,
                    )
                ax.invert_yaxis()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.suptitle(f"Generated images for {self.runname} for epoch: {epoch}")
            plt.show()


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
                generated_data = self.model(batch_size) # handles inverse gumbel in __call__

                def compute_chi_diff(i):
                    generated_data_i = generated_data[..., i]
                    extcoeff = pairwise_extremal_coeffs(generated_data_i)
                    extcoeff_data = getattr(self, f"extcoeffs_{name}_channel_{i}")
                    return tf.sqrt(tf.reduce_mean(tf.square(extcoeff - extcoeff_data)))

                c = tf.shape(data)[-1]
                chi_diffs = tf.map_fn(compute_chi_diff, tf.range(c), dtype=tf.float32)
                rmse = tf.reduce_mean(chi_diffs)
                setattr(self, f"chi_score_{name}", rmse)
                logs[f"chi_score_{name}"] = rmse


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
