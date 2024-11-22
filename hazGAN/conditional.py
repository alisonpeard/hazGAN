"""
Conditional Wasserstein GAN with gradient penalty (cWGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from inspect import signature

from .extreme_value_theory import chi_loss, inv_gumbel
from .tf_utils import DiffAugment
from .unconditional import process_optimizer_kwargs

# %%
def compile_wgan(config, nchannels=2):
    kwargs = process_optimizer_kwargs(config)
    optimizer = getattr(optimizers, config['optimizer'])
    d_optimizer = optimizer(**kwargs)
    g_optimizer = optimizer(**kwargs)
    wgan = WGANGP(config, nchannels=nchannels)
    wgan.compile(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer
        )
    return wgan


def printv(message, verbose):
    if verbose:
        tf.print(message)


# G(z)
def define_generator(config, nchannels=2):
    """
    >>> generator = define_generator(config)
    """
    # input
    z = tf.keras.Input(shape=(config['latent_dims'],), name='noise_input', dtype='float32')
    condition = tf.keras.Input(shape=(1,), name='condition', dtype='float32')
    label = tf.keras.Input(shape=(1,), name="label", dtype='int32')

    # resize label and condition
    label_embedded = layers.Embedding(config['nconditions'], config['embedding_depth'])(label)
    label_embedded = layers.Reshape((config['embedding_depth'],))(label_embedded)
    condition_projected = layers.Dense(config['embedding_depth'], input_shape=(1,))(condition)
    concatenated = layers.concatenate([z, condition_projected, label_embedded])

    # Fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config["g_layers"][0] * 5 * 5 * nchannels, use_bias=False)(concatenated)
    fc = layers.Reshape((5, 5, int(nchannels * config["g_layers"][0])))(fc)
    lrelu0 = layers.LeakyReLU(config['lrelu'])(fc)
    drop0 = layers.Dropout(config['dropout'])(lrelu0)
    if config['normalize_generator']:
        bn0 = layers.BatchNormalization(axis=-1)(drop0)  # normalise along features layer (1024)
    else:
        bn0 = drop0
    
    # 1st deconvolution block, 5 x 5 x 1024 -> 7 x 7 x 512
    conv1 = layers.Conv2DTranspose(config["g_layers"][1], 3, 1, use_bias=False)(bn0)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)
    if config['normalize_generator']:
        bn1 = layers.BatchNormalization(axis=-1)(drop1)
    else:
        bn1 = drop1

    # 2nd deconvolution block, 6 x 8 x 512 -> 14 x 18 x 256
    conv2 = layers.Conv2DTranspose(config["g_layers"][2], (3, 4), 1, use_bias=False)(bn1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)
    if config['normalize_generator']:
        bn2 = layers.BatchNormalization(axis=-1)(drop2)
    else:
        bn2 = drop2

    # Output layer, 17 x 21 x 128 -> 20 x 24 x nchannels, resizing not inverse conv
    conv3 = layers.Resizing(20, 24, interpolation=config['interpolation'])(bn2)
    score = layers.Conv2DTranspose(nchannels, (4, 6), 1, padding='same')(conv3)
    o = score if config['gumbel'] else tf.keras.activations.sigmoid(score) # NOTE: check
    return tf.keras.Model([z, condition, label], o, name="generator")


# D(x)
def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """
    # inputs
    x = tf.keras.Input(shape=(20, 24, nchannels), name='samples')
    condition = tf.keras.Input(shape=(1,), name='condition')
    label = tf.keras.Input(shape=(1,), name='label')

    label_embedded = layers.Embedding(config['nconditions'], config['embedding_depth'] * 20 * 24)(label)
    label_embedded = layers.Reshape((20, 24, config['embedding_depth']))(label_embedded)

    condition_projected = layers.Dense(config['embedding_depth'] * 20 * 24)(condition)
    condition_projected = layers.Reshape((20, 24, config['embedding_depth']))(condition_projected)

    concatenated = layers.concatenate([x, condition_projected, label_embedded])

    # 1st hidden layer 9x10x64
    conv1 = layers.Conv2D(config["d_layers"][0], (4, 5), (2, 2), "valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform())(concatenated)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)

    # 2nd hidden layer 7x7x128
    conv1 = layers.Conv2D(config["d_layers"][1], (3, 4), (1, 1), "valid")(drop1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)

    # 3rd hidden layer 5x5x256
    conv2 = layers.Conv2D(config["d_layers"][2], (3, 3), (1, 1), "valid")(drop2)
    lrelu3 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop3 = layers.Dropout(config['dropout'])(lrelu3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * config["d_layers"][2]))(drop3)
    score = layers.Dense(1)(flat)
    out = layers.Reshape((1,))(score)
    return tf.keras.Model([x, condition, label], out, name="critic")


class WGANGP(keras.Model):
    def __init__(self, config, nchannels=2):
        super().__init__()
        self.critic = define_critic(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config['latent_dims']
        self.lambda_chi = config['lambda_chi']
        self.lambda_gp = config['lambda_gp']
        self.config = config
        self.latent_space_distn = getattr(tf.random, config['latent_space_distn'])
        self.trainable_vars = [
            *self.generator.trainable_variables,
            *self.critic.trainable_variables,
        ]
        if config['gumbel']:
            self.inv = inv_gumbel
        else:
            self.inv = lambda x: x
        self.augment = lambda x: DiffAugment(x, config['augment_policy'])
        self.penalty = config['penalty']

        # trackers average over batches
        self.chi_rmse_tracker = keras.metrics.Mean(name="chi_rmse")
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.critic_loss_tracker = keras.metrics.Mean(name="critic_loss")
        self.value_function_tracker = keras.metrics.Mean(name="value_function")
        self.critic_real_tracker = keras.metrics.Mean(name="critic_real")
        self.critic_fake_tracker = keras.metrics.Mean(name="critic_fake")
        self.seed = config['seed']
        self.critic_steps = 0
        
    
    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_optimizer.build(self.critic.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)


    def call(self, condition, label, nsamples=5,
             latent_vectors=None, temp=1., offset=0, seed=None):
        """Return uniformly distributed samples from the generator."""
        if latent_vectors is None:
            latent_vectors = self.latent_space_distn(
                (nsamples, self.latent_dim),
                temperature=temp,
                offset=offset,
                seed=seed
                )
        else:
            n = latent_vectors.shape[0]
            assert n == nsamples, f"Latent vector must be same length ({n}) as requested number of samples ({nsamples})."

        raw = self.generator([latent_vectors, condition, label], training=False)
        return self.inv(raw)

        
    @tf.function
    def train_step(self, batch):
        data = batch['uniform']
        condition = batch['condition']
        label = batch['label']
        batch_size = tf.shape(data)[0]
        
        # train critic
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator([random_latent_vectors, condition, label], training=False)
        with tf.GradientTape() as tape:
            score_real = self.critic([self.augment(data), condition, label])
            score_fake = self.critic([self.augment(fake_data), condition, label])
            critic_loss = tf.reduce_mean(score_fake) - tf.reduce_mean(score_real) # value function (observed to correlate with sample quality --Gulrajani 2017)
            eps = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
            differences = fake_data - data
            interpolates = data + (eps * differences)  # interpolated data
            with tf.GradientTape() as tape_gp:
                tape_gp.watch(interpolates)
                score = self.critic([interpolates, condition, label])
            gradients = tape_gp.gradient(score, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1])) # NOTE: previously , axis=[1, 2, 3] but Gulrajani code has [1]
            if self.penalty == 'lipschitz':
                gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2) # https://openreview.net/forum?id=B1hYRMbCW
            elif self.penalty == 'gp':
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            else:
                raise ValueError("Penalty must be either 'lipschitz' or 'gp'.")
            critic_loss += self.lambda_gp * gradient_penalty
        grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        self.critic_steps += 1

        # train generator (every n steps)
        if self.critic_steps % self.config['training_balance'] == 0:
            random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                generated_data = self.generator([random_latent_vectors, condition, label])
                score = self.critic(self.augment(generated_data), training=False)
                generator_loss = -tf.reduce_mean(score)
                chi_rmse = chi_loss(self.inv(data), self.inv(generated_data)) # think this is safe inside GradientTape
            grads = tape.gradient(generator_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # update metrics and return their values
            self.critic_loss_tracker(critic_loss)
            self.generator_loss_tracker.update_state(generator_loss)
            self.chi_rmse_tracker.update_state(chi_rmse)
            self.value_function_tracker.update_state(-critic_loss)

        return {
            "chi_rmse": self.chi_rmse_tracker.result(),
            "generator_loss": self.generator_loss_tracker.result(),
            "critic_loss": self.critic_loss_tracker.result(),
            "value_function": self.value_function_tracker.result(),
            'critic_real': tf.reduce_mean(score_real),
            'critic_fake': tf.reduce_mean(score_fake),
        }
