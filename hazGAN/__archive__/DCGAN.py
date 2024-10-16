import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.nn import sigmoid_cross_entropy_with_logits as cross_entropy
from .extreme_value_theory import chi_loss


def process_adam_from_config(config):
    kwargs = {
        "learning_rate": config.learning_rate,
        "beta_1": config.beta_1,
        "beta_2": config.beta_2,
        "use_ema": config.use_ema,
        "ema_momentum": config.ema_momentum,
        "ema_overwrite_frequency": config.ema_overwrite_frequency,
    }
    return kwargs


def compile_dcgan(config, loss_fn=cross_entropy, nchannels=2):
    adam_kwargs = process_adam_from_config(config)
    d_optimizer = Adam(**adam_kwargs)
    g_optimizer = Adam(**adam_kwargs)
    dcgan = DCGAN(config, nchannels=nchannels)
    dcgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, loss_fn=loss_fn)
    return dcgan


# G(z)
def define_generator(config, nchannels=2):
    """
    >>> generator = define_generator()
    """
    z = tf.keras.Input(shape=(100))

    # First fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config["complexity_0"] * config["g_layers"][0])(z)
    fc = layers.Reshape(
        (5, 5, int(config["complexity_0"] * config["g_layers"][0] / 25))
    )(fc)
    fc = layers.BatchNormalization(axis=-1)(fc)  # normalise along features layer (1024)
    lrelu0 = layers.LeakyReLU(config.lrelu)(fc)
    drop0 = layers.Dropout(config.dropout)(lrelu0)

    # Deconvolution, 7 x 7 x 512
    conv1 = tf.keras.layers.Resizing(7, 7, interpolation="nearest")(drop0)
    conv1 = layers.Conv2D(
        config["complexity_1"] * config["g_layers"][1], (3, 3), (1, 1), padding="same", use_bias=False
    )(drop0)
    conv1 = layers.BatchNormalization(axis=-1)(conv1)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)

    # Deconvolution, 9 x 10 x 256
    conv2 = tf.keras.layers.Resizing(9, 10, interpolation="nearest")(drop1)
    conv2 = layers.Conv2D(
        config["complexity_2"] * config["g_layers"][2], (3, 4), (1, 1), padding="same", use_bias=False
    )(drop1)
    conv2 = layers.BatchNormalization(axis=-1)(conv2)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv2)
    drop2 = layers.Dropout(config.dropout)(lrelu2)

    # Output layer, 20 x 24 x nchannels
    conv3 = tf.keras.layers.Resizing(20, 24, interpolation="nearest")(drop2)
    logits = layers.Conv2D(nchannels, (4, 6), (1, 1), padding="same")(conv3)
    o = tf.keras.activations.sigmoid(logits)  # not done in original code but doesn't make sense not to
    return tf.keras.Model(z, o, name="generator")


# D(x)
def define_discriminator(config, nchannels=2):
    """
    >>> discriminator = define_discriminator()
    """
    x = tf.keras.Input(shape=(20, 24, nchannels))

    # 1st hidden layer 9x10x64
    conv1 = layers.Conv2D(
        config["d_layers"][0],
        (4, 5),
        (2, 2),
        "valid",
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(x)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)

    # 2nd hidden layer 7x7x128
    conv1 = layers.Conv2D(
        config["d_layers"][1], (3, 4), (1, 1), "valid", use_bias=False
    )(drop1)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv1)
    drop2 = layers.Dropout(config.dropout)(lrelu2)
    norm2 = layers.BatchNormalization(axis=-1)(drop2)  # turns out often works better AFTER activation

    # 3rd hidden layer 5x5x256
    conv2 = layers.Conv2D(
        config["d_layers"][2], (3, 3), (1, 1), "valid", use_bias=False
    )(norm2)
    lrelu3 = layers.LeakyReLU(config.lrelu)(conv2)
    drop3 = layers.Dropout(config.dropout)(lrelu3)
    norm3 = layers.BatchNormalization(axis=-1)(drop3)

    # fully connected 1x1
    flat = layers.Reshape((-1, 5 * 5 * config["d_layers"][2]))(norm3)
    logits = layers.Dense(1)(flat)
    logits = layers.Reshape((1,))(logits)
    o = tf.keras.activations.sigmoid(logits)

    return tf.keras.Model(x, [o, logits], name="discriminator")


class DCGAN(keras.Model):
    def __init__(self, config, nchannels=2):
        super().__init__()
        self.discriminator = define_discriminator(config, nchannels)
        self.critic = self.discriminator # so can use interchangeably
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config.latent_dims
        self.lambda_ = config.lambda_
        self.config = config
        self.trainable_vars = [
            *self.generator.trainable_variables,
            *self.discriminator.trainable_variables,
        ]
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss_real")
        self.g_loss_raw_tracker = keras.metrics.Mean(name="g_loss_raw")
        self.g_penalty_tracker = keras.metrics.Mean(name="g_penalty")

        if config.latent_space_distn == "uniform":
            self.latent_space_distn = tf.random.uniform
        elif config.latent_space_distn == "normal":
            self.latent_space_distn = tf.random.normal


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

        self.d_optimizer.build(self.discriminator.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)

    def call(self, nsamples=5):
        random_latent_vectors = self.latent_space_distn((nsamples, self.latent_dim))
        return self.generator(random_latent_vectors, training=False)

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator(random_latent_vectors, training=False)
        labels_real = tf.ones((batch_size, 1)) * self.config.true_label_smooth
        labels_fake = tf.zeros((batch_size, 1))

        # train discriminator
        for _ in range(self.config.training_balance):
            with tf.GradientTape() as tape:
                _, logits_real = self.discriminator(data)
                _, logits_fake = self.discriminator(fake_data)
                d_loss_real = self.loss_fn(labels_real, logits_real)
                d_loss_fake = self.loss_fn(labels_fake, logits_fake)
                d_loss = d_loss_real + d_loss_fake
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        # sample random points in the latent space (again)
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        misleading_labels = tf.ones(
            (batch_size, 1)
        )  # i.e., want to trick discriminator

        # train the generator (don't update disciminator weights this time)
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors)
            _, logits = self.discriminator(generated_data, training=False)
            g_loss_raw = tf.reduce_mean(self.loss_fn(misleading_labels, logits))
            g_penalty = self.lambda_ * chi_loss(data, generated_data) # extremal correlation structure
            g_loss = g_loss_raw  #+ g_penalty
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics and return their values
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_raw_tracker.update_state(g_loss_raw)
        self.g_penalty_tracker.update_state(g_penalty)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss_raw": self.g_loss_raw_tracker.result(),
            "g_penalty": self.g_penalty_tracker.result(),
        }
