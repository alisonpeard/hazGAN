"""
Conditional Wasserstein GAN with gradient penalty (cWGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
# %%
import sys
import warnings
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from inspect import signature

from .extreme_value_theory import chi_loss, inv_gumbel
from .tf_utils import DiffAugment
from .tf_utils import wrappers

# %%
def get_optimizer_kwargs(optimizer):
    optimizer = getattr(optimizers, optimizer)
    params = signature(optimizer).parameters
    return params


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
    
    return kwargs


def compile_wgan(config, nchannels=2):
    kwargs = process_optimizer_kwargs(config)
    optimizer = getattr(optimizers, config['optimizer'])
    critic_optimizer = optimizer(**kwargs)
    generator_optimizer = optimizer(**kwargs)
    wgan = WGANGP(config, nchannels=nchannels)
    wgan.compile(
        critic_optimizer=critic_optimizer,
        generator_optimizer=generator_optimizer
        )
    return wgan


def printv(message, verbose):
    if verbose:
        tf.print(message)


def define_generator(config, nchannels=2):
    """
    >>> generator = define_generator(config)
    """
    # input
    z = tf.keras.Input(shape=(config['latent_dims'],), name='noise_input', dtype='float32')
    condition = tf.keras.Input(shape=(1,), name='condition', dtype='float32')
    label = tf.keras.Input(shape=(1,), name="label", dtype='int32')

    # resize label and condition
    label_embedded = wrappers.Embedding(config['nconditions'], config['embedding_depth'])(label)
    label_embedded = layers.Reshape((config['embedding_depth'],))(label_embedded)
    condition_projected = wrappers.Dense(config['embedding_depth'], input_shape=(1,))(condition)
    concatenated = layers.concatenate([z, condition_projected, label_embedded])

    # Fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = wrappers.Dense(config["g_layers"][0] * 5 * 5 * nchannels, use_bias=False)(concatenated)
    fc = layers.Reshape((5, 5, int(nchannels * config["g_layers"][0])))(fc)
    lrelu0 = layers.LeakyReLU(config['lrelu'])(fc)
    drop0 = layers.Dropout(config['dropout'])(lrelu0)
    if config['normalize_generator']:
        bn0 = layers.BatchNormalization(axis=-1)(drop0)  # normalise along features layer (1024)
    else:
        bn0 = drop0
    
    # 1st deconvolution block, 5 x 5 x 1024 -> 7 x 7 x 512
    conv1 = wrappers.Conv2DTranspose(config["g_layers"][1], 3, 1, use_bias=False)(bn0)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)
    if config['normalize_generator']:
        bn1 = layers.BatchNormalization(axis=-1)(drop1)
    else:
        bn1 = drop1

    # 2nd deconvolution block, 6 x 8 x 512 -> 14 x 18 x 256
    conv2 = wrappers.Conv2DTranspose(config["g_layers"][2], (3, 4), 1, use_bias=False)(bn1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)
    if config['normalize_generator']:
        bn2 = layers.BatchNormalization(axis=-1)(drop2)
    else:
        bn2 = drop2

    # Output layer, 17 x 21 x 128 -> 20 x 24 x nchannels, resizing not inverse conv
    conv3 = layers.Resizing(20, 24, interpolation=config['interpolation'])(bn2)
    score = wrappers.Conv2DTranspose(nchannels, (4, 6), 1, padding='same')(conv3)
    o = score if config['gumbel'] else tf.keras.activations.sigmoid(score) # NOTE: check
    return tf.keras.Model([z, condition, label], o, name="generator")


def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """
    # inputs
    x = tf.keras.Input(shape=(20, 24, nchannels), name='samples')
    condition = tf.keras.Input(shape=(1,), name='condition')
    label = tf.keras.Input(shape=(1,), name='label')

    label_embedded = wrappers.Embedding(config['nconditions'], config['embedding_depth'] * 20 * 24)(label)
    label_embedded = layers.Reshape((20, 24, config['embedding_depth']))(label_embedded)

    condition_projected = wrappers.Dense(config['embedding_depth'] * 20 * 24)(condition)
    condition_projected = layers.Reshape((20, 24, config['embedding_depth']))(condition_projected)

    concatenated = layers.concatenate([x, condition_projected, label_embedded])

    # 1st hidden layer 9x10x64
    conv1 = wrappers.Conv2D(config["d_layers"][0], (4, 5), (2, 2), "valid")(concatenated)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)

    # 2nd hidden layer 7x7x128
    conv1 = wrappers.Conv2D(config["d_layers"][1], (3, 4), (1, 1), "valid")(drop1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)

    # 3rd hidden layer 5x5x256
    conv2 = wrappers.Conv2D(config["d_layers"][2], (3, 3), (1, 1), "valid")(drop2)
    lrelu3 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop3 = layers.Dropout(config['dropout'])(lrelu3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * config["d_layers"][2]))(drop3)
    score = wrappers.Dense(1)(flat) #? sigmoid might smooth training by constraining?, S did similar, caused nans
    out = layers.Reshape((1,))(score)
    return tf.keras.Model([x, condition, label], out, name="critic")


class WGANGP(keras.Model):
    def __init__(self, config, nchannels=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.critic = define_critic(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config['latent_dims']
        self.lambda_gp = config['lambda_gp']
        self.lambda_condition = config['lambda_condition']
        self.config = config
        self.latent_space_distn = getattr(tf.random, config['latent_space_distn'])
        self.trainable_vars = [
            *self.generator.trainable_variables,
            *self.critic.trainable_variables,
        ]
        if config['gumbel']:
            self.inv = inv_gumbel
        else:
            self.inv = lambda x: x # will make serialising etc. difficult
        self.augment = lambda x: DiffAugment(x, config['augment_policy'])
        self.seed = config['seed']

        # stateful metrics
        self.chi_rmse_tracker = keras.metrics.Mean(name="chi_rmse")
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.critic_loss_tracker = keras.metrics.Mean(name="critic_loss")
        self.value_function_tracker = keras.metrics.Mean(name="value_function")
        self.critic_real_tracker = keras.metrics.Mean(name="critic_real")
        self.critic_fake_tracker = keras.metrics.Mean(name="critic_fake")
        self.critic_valid_tracker = keras.metrics.Mean(name="critic_valid")

        # training statistics # ? setting dtype=tf.int32 fails ?
        self.images_seen = keras.metrics.Sum(name="images_seen")
        self.critic_steps = keras.metrics.Sum(name="critic_steps")
        self.generator_steps = keras.metrics.Sum(name="generator_steps")
        self.critic_grad_norm = keras.metrics.Mean(name="critic_grad_norm")
        self.generator_grad_norm = keras.metrics.Mean(name="generator_grad_norm")

        # monitor for vanishing gradients
        self.critic_grad_norms = [
            keras.metrics.Mean(name=f"critic_{i}_{var.path}") for i, var in enumerate(self.critic.trainable_variables)
        ]
        self.generator_grad_norms = [
            keras.metrics.Mean(name=f"generator_{i}_{var.path}") for i, var in enumerate(self.generator.trainable_variables)
        ]

    def compile(self, critic_optimizer, generator_optimizer, *args, **kwargs) -> None:
        super().compile(*args, **kwargs)
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer.build(self.critic.trainable_variables)
        self.generator_optimizer.build(self.generator.trainable_variables)


    def call(self, condition, label, nsamples=5,
             latent_vectors=None, temp=1., offset=0, seed=None) -> tf.Tensor:
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
    

    def evaluate(self, x, **kwargs) -> dict:
        """Overwrite evaluation function for custom data.
        """
        score_valid = 0
        with warnings.catch_warnings(): # suppress out of range error
            warnings.filterwarnings("ignore", message="Local rendezvous")
            for n, batch in enumerate(x):
                try:
                    data = batch['uniform']
                    condition = batch["condition"]
                    label = batch["label"]
                    critic_score = self.critic([data, condition, label], training=False)
                    score_valid += tf.reduce_mean(critic_score)
                except tf.errors.OutOfRangeError:
                    break
        score_valid = score_valid / (n + 1)
        self.critic_valid_tracker.update_state(score_valid)
        return {'critic': self.critic_valid_tracker.result()}


    def train_critic(self, data, condition, label, batch_size) -> None:
        """Train critic with gradient penalty."""
        print("\nTracing critic...")
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator([random_latent_vectors, condition, label], training=False)

        with tf.GradientTape() as tape:
            score_real = self.critic([self.augment(data), condition, label])
            score_fake = self.critic([self.augment(fake_data), condition, label])
            critic_loss = tf.reduce_mean(score_fake) - tf.reduce_mean(score_real) # value function (observed to correlate with sample quality --Gulrajani 2017)
            eps = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
            differences = fake_data - data
            interpolates = data + (eps * differences)  # interpolated data

            with tf.GradientTape() as tape_gp:
                tape_gp.watch(interpolates)
                score = self.critic([interpolates, condition, label])
            gradients = tape_gp.gradient(score, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            print("Shape of interpolated gradients:", slopes.shape)

            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            critic_loss += self.lambda_gp * gradient_penalty

        gradients = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_weights))

        self.critic_loss_tracker(critic_loss)
        self.value_function_tracker.update_state(-critic_loss)
        self.critic_real_tracker.update_state(tf.reduce_mean(score_real))
        self.critic_fake_tracker.update_state(tf.reduce_mean(score_fake))

        self.critic_steps.update_state(tf.constant(1, dtype=tf.int32))

        # get gradient norms
        global_gradient_norm = tf.linalg.global_norm(gradients)
        self.critic_grad_norm.update_state(global_gradient_norm)

        for i, grad in enumerate(gradients):
            self.critic_grad_norms[i].update_state(tf.linalg.global_norm([grad]))

        return None

    
    def train_generator(self, data, condition, label, batch_size) -> None:
        """https://www.tensorflow.org/guide/function#conditionals
        """
        print("\nTracing generator...\n")
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_data = self.generator([random_latent_vectors, condition, label])
            score = self.critic([self.augment(generated_data), condition, label], training=False)
            generator_loss = -tf.reduce_mean(score)
            condition_penalty = tf.reduce_mean(tf.square(tf.reduce_max(generated_data[..., 0], axis=[1, 2]) - condition))
            generator_penalised_loss = generator_loss #!+ self.lambda_condition * condition_penalty
        
        chi_rmse = chi_loss(self.inv(data), self.inv(generated_data))
        gradients = tape.gradient(generator_penalised_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))
        
        self.generator_loss_tracker.update_state(generator_loss)
        self.chi_rmse_tracker.update_state(chi_rmse)

        self.generator_steps.update_state(tf.constant(1, dtype=tf.int32))

        # get gradient norms
        global_gradient_norm = tf.linalg.global_norm(gradients)
        self.generator_grad_norm.update_state(global_gradient_norm)
        
        for i, grad in enumerate(gradients):
            self.generator_grad_norms[i].update_state(tf.linalg.global_norm([grad]))

        return None


    def skip(self, *args, **kwargs) -> None:
        return None
    

    @tf.function
    def train_step(self, batch) -> dict:
        """print(train_step.pretty_printed_concrete_signatures())"""
        data = batch['uniform']
        condition = batch['condition']
        label = batch['label']
        batch_size = tf.shape(data)[0] # dynamic for graph mode
        
        # train critic
        self.train_critic(data, condition, label, batch_size)

        metrics = {
            "critic_loss": self.critic_loss_tracker.result(),
            "value_function": -self.value_function_tracker.result(),
            'critic_real': self.critic_real_tracker.result(),
            'critic_fake': self.critic_fake_tracker.result()
        }

        # train generator
        generator_flag = tf.math.logical_or(tf.math.equal(
            self.critic_steps.result(), 1),
            tf.math.equal(self.critic_steps.result() % self.config['training_balance'], 0)
            )
        train_generator = lambda: self.train_generator(data, condition, label, batch_size)
        skip_generator = lambda: self.skip()
        tf.cond(
            generator_flag,
            train_generator,
            skip_generator
            )
        
        # update metrics
        self.images_seen.update_state(batch_size)
        metrics["generator_loss"] = self.generator_loss_tracker.result()
        metrics['chi_rmse'] = self.chi_rmse_tracker.result()
        metrics['critic_steps'] = self.critic_steps.result()
        metrics['generator_steps'] = self.generator_steps.result()

        metrics["images_seen"] = self.images_seen.result()
        metrics["critic_grad_norm"] = self.critic_grad_norm.result()
        metrics["generator_grad_norm"] = self.generator_grad_norm.result()
        
        i = 0
        for var, metric in zip(self.critic.trainable_variables, self.critic_grad_norms):
            metrics[f"critic_{i}_{var.path}"] = metric.result()
            i += 1

        i = 0
        for var, metric in zip(self.generator.trainable_variables, self.generator_grad_norms):
            metrics[f"generator_{i}_{var.path}"] = metric.result()
            i += 1

        # print logs if in eager mode
        if tf.executing_eagerly():
            print(f"\nBatch mean:", tf.math.reduce_mean(data))
            print(f"Batch std:", tf.math.reduce_std(data))
        
        return metrics

    @property
    def metrics(self) -> list:
        """Define which stateful metrics to reset per-epoch."""
        return [
            self.critic_real_tracker,
            self.critic_fake_tracker,
            self.critic_valid_tracker,
            self.critic_loss_tracker,
            self.generator_loss_tracker,
            self.value_function_tracker,
            self.chi_rmse_tracker,
            self.critic_grad_norm,
            self.generator_grad_norm
        ] + self.critic_grad_norms + self.generator_grad_norms

# %%
