"""
Conditional Wasserstein GAN with gradient penalty (cWGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
# %%
import warnings
import functools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
# from tensorflow.keras.optimizers.schedules import ExponentialDecay # will use again

import tracemalloc
from inspect import signature
from memory_profiler import profile

from .statistics import chi_loss, inv_gumbel
from .tensorflow import DiffAugment
from .tensorflow import wrappers

logfile = open("trainstep.log", 'w+')

# %%
def sample_gumbel(shape, eps=1e-20, temperature=1., offset=0., seed=None):
    """Sample from Gumbel(0, 1)"""
    T = tf.constant(temperature, dtype=tf.float32)
    O = tf.constant(offset, dtype=tf.float32)
    U = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
    return O - T * tf.math.log(-tf.math.log(U + eps) + eps)


def initialise_variables():
    """Initialise mutable Tensorflow objects."""
    tf.random.gumbel = sample_gumbel


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
    initialise_variables()
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

    def normalise(x):
        if config['normalize_generator']:
            return layers.BatchNormalization(axis=-1)(x)
        else:
            return x
        
    width = config['generator_width']

    assert width % 8 == 0, "generator width must be divisible by 8"
    assert width >= 64, "generator width must be at least 64"

    width0 = width
    width1 = width // 2
    width2 = width // 3
    
    # flexible input processing
    z = tf.keras.Input(shape=(config['latent_dims'],), name='noise_input', dtype='float32')
    inputs = [z] #TODO: list accumulation may not work as expected with AutoGraph
    if config['condition']:
        condition = tf.keras.Input(shape=(1,), name='condition', dtype='float32')
        condition_projected = wrappers.Dense(config['embedding_depth'], input_shape=(1,))(condition)
        condition_projected = layers.LeakyReLU(config['lrelu'])(condition_projected)
        condition_projected = wrappers.Dense(config['embedding_depth'])(condition_projected) # add complexity
        inputs.append(condition_projected) # is this okay in AutoGraph?
    if config['labels']:
        label = tf.keras.Input(shape=(1,), name="label", dtype='int32')
        label_embedded = wrappers.Embedding(config['nconditions'], config['embedding_depth'])(label)
        label_embedded = layers.Reshape((config['embedding_depth'],))(label_embedded)
        inputs.append(label_embedded)
    concatenated = layers.concatenate(inputs)

    # Fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = wrappers.Dense(width0 * 5 * 5 * nchannels, use_bias=False)(concatenated)
    fc = layers.Reshape((5, 5, int(nchannels * width0)))(fc)
    lrelu0 = layers.LeakyReLU(config['lrelu'])(fc)
    drop0 = layers.Dropout(config['dropout'])(lrelu0)
    bn0 = normalise(drop0)
    
    # 1st deconvolution block, 5 x 5 x 1024 -> 7 x 7 x 512
    conv1 = wrappers.Conv2DTranspose(width1, 3, 1, use_bias=False)(bn0)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)
    bn1 = normalise(drop1)

    # 2nd deconvolution block, 6 x 8 x 512 -> 14 x 18 x 256
    conv2 = wrappers.Conv2DTranspose(width2, (3, 4), 1, use_bias=False)(bn1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)
    bn2 = normalise(drop2)

    # Output layer, 17 x 21 x 128 -> 20 x 24 x nchannels, resizing not inverse conv
    conv3 = layers.Resizing(20, 24, interpolation=config['interpolation'])(bn2)
    score = wrappers.Conv2DTranspose(nchannels, (4, 6), 1, padding='same')(conv3)

    # this is new: should control output range better
    if config['gumbel']:
        o = wrappers.GumbelIsh()(score)
        # o = score
    else:
        o = tf.keras.activations.sigmoid(score)
    
    return tf.keras.Model([z, condition, label], o, name="generator")


def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """

    def normalise(x):
        if config['normalize_critic']:
            return layers.LayerNormalization(axis=-1)(x)
        else:
            return x
        
    width = config['critic_width']
    assert config['critic_width'] % 8 == 0, "critic width must be divisible by 8."
    assert config['critic_width'] >= 64, "critic width must be at least 64." 
    
    width2 = width
    width1 = width2 // 2
    width0 = width1 // 2
        
    # flexible input processsing
    x = tf.keras.Input(shape=(20, 24, nchannels), name='samples')
    inputs = [x]
    if config['condition']:
        condition = tf.keras.Input(shape=(1,), name='condition')
        condition_projected = wrappers.Dense(config['embedding_depth'] * 20 * 24)(condition)
        condition_projected = layers.LeakyReLU(config['lrelu'])(condition_projected)
        condition_projected = wrappers.Dense(config['embedding_depth'] * 20 * 24)(condition_projected)
        condition_projected = layers.Reshape((20, 24, config['embedding_depth']))(condition_projected)
        inputs.append(condition_projected)
    if config['labels']:
        label = tf.keras.Input(shape=(1,), name='label')
        label_embedded = wrappers.Embedding(config['nconditions'], config['embedding_depth'] * 20 * 24)(label)
        label_embedded = layers.Reshape((20, 24, config['embedding_depth']))(label_embedded)
        inputs.append(label_embedded)
    concatenated = layers.concatenate(inputs)

    # 1st hidden layer 9x10x64
    conv1 = wrappers.Conv2D(width0, (4, 5), (2, 2), "valid")(concatenated)
    lrelu1 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop1 = layers.Dropout(config['dropout'])(lrelu1)
    drop1 = normalise(drop1)

    # 2nd hidden layer 7x7x128
    conv1 = wrappers.Conv2D(width1, (3, 4), (1, 1), "valid")(drop1)
    lrelu2 = layers.LeakyReLU(config['lrelu'])(conv1)
    drop2 = layers.Dropout(config['dropout'])(lrelu2)
    drop2 = normalise(drop2)

    # 3rd hidden layer 5x5x256
    conv2 = wrappers.Conv2D(width2, (3, 3), (1, 1), "valid")(drop2)
    lrelu3 = layers.LeakyReLU(config['lrelu'])(conv2)
    drop3 = layers.Dropout(config['dropout'])(lrelu3)
    drop3 = normalise(drop3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * width2))(drop3)
    score = wrappers.Dense(1)(flat) #? sigmoid might smooth training by constraining?, S did similar, caused nans
    out = layers.Reshape((1,))(score)
    return tf.keras.Model([x, condition, label], out, name="critic")


class WGANGP(keras.Model):
    """Wasserstein GAN with gradient penalty."""

    # this should improve memory usage
    __slots__ = ['critic', 'generator', 'latent_dim', 'lambda_gp',
                    'config', 'latent_space_distn', 'trainable_vars', 'inv', 'augment',
                    'seed', 'chi_rmse_tracker', 'generator_loss_tracker', 'critic_loss_tracker',
                    'value_function_tracker', 'critic_real_tracker', 'critic_fake_tracker',
                    'critic_valid_tracker', 'images_seen', 'critic_steps', 'generator_steps',
                    'critic_grad_norm', 'generator_grad_norm', 'critic_grad_norms',
                    'generator_grad_norms']

    def __init__(self, config, nchannels=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.critic = define_critic(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config['latent_dims']
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
            self.inv = lambda x: x # will make serialising etc. difficult
        self.augment = functools.partial(DiffAugment, policy=config['augment_policy'])
        self.seed = config['seed']

        # stateful metrics
        self.chi_rmse_tracker = keras.metrics.Mean(name="chi_rmse")
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.critic_loss_tracker = keras.metrics.Mean(name="critic_loss")
        self.value_function_tracker = keras.metrics.Mean(name="value_function")
        self.critic_real_tracker = keras.metrics.Mean(name="critic_real")
        self.critic_fake_tracker = keras.metrics.Mean(name="critic_fake")
        self.critic_valid_tracker = keras.metrics.Mean(name="critic_valid")
        self.gradient_penalty_tracker = keras.metrics.Mean(name="gradient_penalty")

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
             noise=None, temp=1., offset=0, seed=None) -> tf.Tensor:
        """Return uniformly distributed samples from the generator."""
        if noise is None:
            noise = self.latent_space_distn(
                (nsamples, self.latent_dim),
                temperature=temp,
                offset=offset,
                seed=seed
                )
        else:
            n = noise.shape[0]
            assert n == nsamples, f"Latent vector must be same length ({n}) as requested number of samples ({nsamples})."

        raw = self.generator([noise, condition, label], training=False)
        tf.print("Minimum before transformation:", tf.reduce_min(raw))
        tf.print("Maximum before transformation:", tf.reduce_max(raw))
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
        """Train critic with gradient penalty.
        
        Debugging:
            >> concrete_fn = self.train_step.get_concrete_function(dict(
                uniform=tf.TensorSpec(shape=(None,20,24,2)),
                condition=tf.TensorSpec(shape=(None,)),
                label=tf.TensorSpec(shape=(None,))
                ))
            >> print(self.train_step.pretty_printed_concrete_signatures())
            >> graph = concrete_fn.graph
            >> for node in graph.as_graph_def().node:
                    print(f'{node.input} -> {node.name}')
        """
        print("\nTracing critic...")
        random_noise = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator([random_noise, condition, label], training=False)

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
        self.gradient_penalty_tracker.update_state(gradient_penalty)

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
        random_noise = self.latent_space_distn((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_data = self.generator([random_noise, condition, label])
            score = self.critic([self.augment(generated_data), condition, label], training=False)
            generator_loss = -tf.reduce_mean(score)
            generator_penalised_loss = generator_loss
        
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
    
    @profile(stream=logfile)
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
            'critic_fake': self.critic_fake_tracker.result(),
            'gradient_penalty': self.gradient_penalty_tracker.result()
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
