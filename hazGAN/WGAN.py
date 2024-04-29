"""
Wasserstein GAN with gradient penalty (WGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from inspect import signature
from .extreme_value_theory import chi_loss, inv_gumbel


def get_optimizer_kwargs(optimizer):
    optimizer = getattr(optimizers, optimizer)
    params = signature(optimizer).parameters
    return params


def process_optimizer_kwargs(config):
    kwargs = {
        "learning_rate": config.learning_rate,
        "beta_1": config.beta_1,
        "beta_2": config.beta_2,
        "use_ema": config.use_ema,
        "ema_momentum": config.ema_momentum,
        "ema_overwrite_frequency": config.ema_overwrite_frequency,
    }
    params = get_optimizer_kwargs(config.optimizer)
    kwargs = {key: val for key, val in kwargs.items() if key in params}
    return kwargs


def compile_wgan(config, nchannels=2):
    kwargs = process_optimizer_kwargs(config)
    optimizer = getattr(optimizers, config.optimizer)
    d_optimizer = optimizer(**kwargs)
    g_optimizer = optimizer(**kwargs)
    wgan = WGAN(config, nchannels=nchannels)
    wgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer)
    return wgan


# G(z)
def define_generator(config, nchannels=2):
    """
    TODO: Get rid of dense layers so its resolutiuon invariant - Harris (2022).
    >>> generator = define_generator()
    """
    z = tf.keras.Input(shape=(100))

    # First fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config["complexity_0"] * config["g_layers"][0])(z)
    fc = layers.Reshape((5, 5, int(config["complexity_0"] * config["g_layers"][0] / 25)))(fc)
    lrelu0 = layers.LeakyReLU(config.lrelu)(fc)
    drop0 = layers.Dropout(config.dropout)(lrelu0)
    bn0 = layers.BatchNormalization(axis=-1)(drop0)  # normalise along features layer (1024)

    # Deconvolution, 5 x 5 x 1024 -> 7 x 7 x 512
    conv1 = layers.Conv2DTranspose(config["complexity_1"] * config["g_layers"][1], 3, 1, use_bias=False)(bn0)
    # conv1 = tf.keras.layers.Resizing(7, 7, interpolation="nearest")(drop0)
    # conv1 = layers.Conv2D(config["complexity_1"] * config["g_layers"][1], (3, 3), (1, 1), padding="same", use_bias=False)(drop0)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)
    bn1 = layers.BatchNormalization(axis=-1)(drop1)

    # Deconvolution, 6 x 8 x 512 -> 14 x 18 x 256
    conv2 = layers.Conv2DTranspose(config["complexity_2"] * config["g_layers"][2], (3, 4), 1, use_bias=False)(bn1)
    # conv2 = tf.keras.layers.Resizing(9, 10, interpolation="nearest")(drop1)
    # conv2 = layers.Conv2D(config["complexity_2"] * config["g_layers"][2], (3, 4), (1, 1), padding="same", use_bias=False)(drop1)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv2)
    drop2 = layers.Dropout(config.dropout)(lrelu2)
    bn2 = layers.BatchNormalization(axis=-1)(drop2)

    # Output layer, 17 x 21 x 128 -> 20 x 24 x nchannels
    score = layers.Conv2DTranspose(nchannels, (4, 6), 2)(bn2)
    # conv3 = tf.keras.layers.Resizing(20, 24, interpolation="nearest")(drop2)
    # score = layers.Conv2D(nchannels, (4, 6), (1, 1), padding="same")(conv3)
    o = score if config.gumbel else tf.keras.activations.sigmoid(score) # NOTE: check
    return tf.keras.Model(z, o, name="generator")


# D(x)
def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """
    x = tf.keras.Input(shape=(20, 24, nchannels))

    # 1st hidden layer 9x10x64
    conv1 = layers.Conv2D(config["d_layers"][0], (4, 5), (2, 2), "valid", kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)

    # 2nd hidden layer 7x7x128
    conv1 = layers.Conv2D(config["d_layers"][1], (3, 4), (1, 1), "valid", use_bias=False)(drop1)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv1)
    drop2 = layers.Dropout(config.dropout)(lrelu2)

    # 3rd hidden layer 5x5x256
    conv2 = layers.Conv2D(config["d_layers"][2], (3, 3), (1, 1), "valid", use_bias=False)(drop2)
    lrelu3 = layers.LeakyReLU(config.lrelu)(conv2)
    drop3 = layers.Dropout(config.dropout)(lrelu3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * config["d_layers"][2]))(drop3)
    score = layers.Dense(1)(flat)
    out = layers.Reshape((1,))(score)
    return tf.keras.Model(x, out, name="critic")


class WGAN(keras.Model):
    def __init__(self, config, nchannels=2):
        super().__init__()
        self.critic = define_critic(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config.latent_dims
        self.lambda_ = config.lambda_
        self.lambda_gp = config.lambda_gp
        self.gumbel = config.gumbel
        self.config = config
        self.latent_space_distn = getattr(tf.random, config.latent_space_distn)
        self.trainable_vars = [
            *self.generator.trainable_variables,
            *self.critic.trainable_variables,
        ]
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_raw_tracker = keras.metrics.Mean(name="g_loss_raw")
        self.g_penalty_tracker = keras.metrics.Mean(name="g_penalty")


    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_optimizer.build(self.critic.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)

    def call(self, nsamples=5):
        random_latent_vectors = self.latent_space_distn((nsamples, self.latent_dim))
        raw = self.generator(random_latent_vectors, training=False)
        if self.gumbel:
            return inv_gumbel(raw)
        else:
            return raw

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator(random_latent_vectors, training=False)

        # train critic
        for _ in range(self.config.training_balance):
            with tf.GradientTape() as tape:
                score_real = self.critic(data)
                score_fake = self.critic(fake_data)
                d_loss = tf.reduce_mean(score_fake) - tf.reduce_mean(score_real)

                # NOTE: https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py
                eps = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                differences = fake_data - data
                interpolates = data + (eps * differences)  # interpolated data
                with tf.GradientTape() as tape_gp:
                    tape_gp.watch(interpolates)
                    score = self.critic(interpolates)
                gradients = tape_gp.gradient(score, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
                d_loss += self.lambda_gp * gradient_penalty

            grads = tape.gradient(d_loss, self.critic.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        # train generator
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors)
            score = self.critic(generated_data, training=False)
            g_loss_raw = -tf.reduce_mean(score)
            #g_penalty = self.lambda_ * chi_loss(data, generated_data) # extremal correlation structure
            g_loss = g_loss_raw #+ g_penalty 
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics and return their values
        self.d_loss_tracker(d_loss)
        self.g_loss_raw_tracker.update_state(g_loss_raw)
        #self.g_penalty_tracker.update_state(g_penalty)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss_raw": self.g_loss_raw_tracker.result(),
            "g_penalty": self.g_penalty_tracker.result(),
        }
