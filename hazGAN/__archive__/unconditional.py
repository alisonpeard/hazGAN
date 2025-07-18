"""
Wasserstein GAN with gradient penalty (WGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
import numpy as np
import pytorch as tf
from pytorch import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from inspect import signature

from .extreme_value_theory import chi_loss, inv_gumbel
from .tf_utils import DiffAugment


def sample_gumbel(shape, eps=1e-20, temperature=1., offset=0., seed=None):
    """Sample from Gumbel(0, 1)"""
    T = tf.constant(temperature, dtype=tf.float32)
    O = tf.constant(offset, dtype=tf.float32)
    U = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
    return O - T * tf.math.log(-tf.math.log(U + eps) + eps)
tf.random.gumbel = sample_gumbel


def get_optimizer_kwargs(optimizer):
    optimizer = getattr(optimizers, optimizer)
    params = signature(optimizer).parameters
    return params


def exponential_decay(config):
    initial_lr = config['learning_rate']
    final_lr = initial_lr / config['final_lr_factor']
    decay_factor = (final_lr / initial_lr) ** (1 / config['epochs'])
    steps_per_epoch = int(config['train_size'] / config['batch_size'])

    schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=steps_per_epoch,
        decay_rate=decay_factor,
        staircase=True
    )
    return schedule

def process_optimizer_kwargs(config):
    kwargs = {
        "learning_rate": config['learning_rate'],
        "beta_1": config['beta_1'],
        "beta_2": config['beta_2'],
        "weight_decay": config['weight_decay'],
        "use_ema": config['use_ema'],
        "ema_momentum": config['ema_momentum'],
        "ema_overwrite_frequency": config['ema_overwrite_frequency'],
    }
    params = get_optimizer_kwargs(config['optimizer'])
    kwargs = {key: val for key, val in kwargs.items() if key in params}

    if config.get('lr_decay', False):
        lr_schedule = exponential_decay(config)
        kwargs["learning_rate"] = lr_schedule
    
    return kwargs


def compile_wgan(config, nchannels=2, train=None):
    kwargs = process_optimizer_kwargs(config)
    optimizer = getattr(optimizers, config['optimizer'])
    d_optimizer = optimizer(**kwargs)
    g_optimizer = optimizer(**kwargs)
    wgan = WGANGP(config, nchannels=nchannels, train=train)
    wgan.compile(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer
        )
    return wgan


# G(z)
def define_generator(config, nchannels=2):
    """
    >>> generator = define_generator()
    """
    z = tf.keras.Input(shape=(config['latent_dims'],))

    # Fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config["g_layers"][0] * 5 * 5 * nchannels, use_bias=False)(z)
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
    return tf.keras.Model(z, o, name="generator")


# D(x)
def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """
    x = tf.keras.Input(shape=(20, 24, nchannels))

    # 1st hidden layer 9x10x64
    conv1 = layers.Conv2D(config["d_layers"][0], (4, 5), (2, 2), "valid",
                          kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
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
    return tf.keras.Model(x, out, name="critic")


class WGANGP(keras.Model):
    def __init__(self, config, nchannels=2, train=None):
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
        self.train = train
        
    
    def compile(self, d_optimizer, g_optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_optimizer.build(self.critic.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)


    def call(self, nsamples=5, temp=1., offset=0, seed=None):
        """Return uniformly distributed samples from the generator."""
        random_latent_vectors = self.latent_space_distn(
            (nsamples, self.latent_dim),
            temperature=temp,
            offset=offset,
            seed=seed
            )
        raw = self.generator(random_latent_vectors, training=False)
        return self.inv(raw)


    def cleanup(self):
        self.train = None
        
    @tf.function
    def train_step(self, data):
        # removing this to sample multiple times from train_generator
        if self.train is None: # single sample per iter
            data = data[0]
            batch_size = tf.shape(data)[0]
            random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
            fake_data = self.generator(random_latent_vectors, training=False)

        # train critic
        # https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py:134
        for _ in range(self.config['training_balance']):
            if self.train is not None: # infinite generation per iter
                data = next(iter(self.train))[0]
                batch_size = tf.shape(data)[0]
                random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
                fake_data = self.generator(random_latent_vectors, training=False)

            with tf.GradientTape() as tape:
                score_real = self.critic(self.augment(data))
                score_fake = self.critic(self.augment(fake_data))
                critic_loss = tf.reduce_mean(score_fake) - tf.reduce_mean(score_real) # value function (observed to correlate with sample quality --Gulrajani 2017)
                eps = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                differences = fake_data - data
                interpolates = data + (eps * differences)  # interpolated data
                with tf.GradientTape() as tape_gp:
                    tape_gp.watch(interpolates)
                    score = self.critic(interpolates)
                gradients = tape_gp.gradient(score, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3])) # NOTE: previously , axis=[1, 2, 3] but Gulrajani code has [1]
                if self.penalty == 'lipschitz':
                    gradient_penalty = tf.reduce_mean(tf.clip_by_value(slopes - 1., 0., np.infty)**2) # https://openreview.net/forum?id=B1hYRMbCW
                elif self.penalty == 'gp':
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                else:
                    raise ValueError("Penalty must be either 'lipschitz' or 'gp'.")
                critic_loss += self.lambda_gp * gradient_penalty

            grads = tape.gradient(critic_loss, self.critic.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        # train generator
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors)
            score = self.critic(self.augment(generated_data), training=False)
            generator_loss_raw = -tf.reduce_mean(score)
            chi_rmse = chi_loss(self.inv(data), self.inv(generated_data)) # think this is safe inside GradientTape
            if self.lambda_chi > 0: # NOTE: this doesn't work with GPU
                generator_loss = generator_loss_raw + (self.lambda_chi * chi_rmse)
            else:
                generator_loss = generator_loss_raw
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics and return their values
        self.critic_loss_tracker(critic_loss)
        self.generator_loss_tracker.update_state(generator_loss_raw)
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
