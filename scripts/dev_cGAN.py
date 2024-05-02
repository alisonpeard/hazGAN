"""
https://keras.io/examples/generative/conditional_gan/
"""
#%%
import os
import hazGAN as hg
import tensorflow as tf
from tensorflow.keras import layers

data_source = "era5"
cwd = os.getcwd()  # scripts directory
wd = os.path.join(cwd, "..")  # hazGAN directory
datadir = os.path.join(wd, "..", f"{data_source}_data")  # keep data folder in parent directory
# %%
# load data
[train_u, test_u], [train_x, test_x], [train_m, test_m], [train_z, test_z], params = hg.load_training(datadir, 1000, 'reflect', gumbel_marginals=True)

train = tf.data.Dataset.from_tensor_slices((train_u, train_z)).batch(10)
test = tf.data.Dataset.from_tensor_slices((test_u, test_z)).batch(10)
# %%
import wandb
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)

def define_generator(config, nchannels=2):
    z = tf.keras.Input(shape=(100 + 1,))

    # First fully connected layer, 1 x 1 x 25600 -> 5 x 5 x 1024
    fc = layers.Dense(config["complexity_0"] * config["g_layers"][0])(z)
    fc = layers.Reshape((5, 5, int(config["complexity_0"] * config["g_layers"][0] / 25)))(fc)
    lrelu0 = layers.LeakyReLU(config.lrelu)(fc)
    drop0 = layers.Dropout(config.dropout)(lrelu0)
    bn0 = layers.BatchNormalization(axis=-1)(drop0)  # normalise along features layer (1024)

    # Deconvolution, 5 x 5 x 1024 -> 7 x 7 x 512
    conv1 = layers.Conv2DTranspose(config["complexity_1"] * config["g_layers"][1], 3, 1, use_bias=False)(bn0)
    lrelu1 = layers.LeakyReLU(config.lrelu)(conv1)
    drop1 = layers.Dropout(config.dropout)(lrelu1)
    bn1 = layers.BatchNormalization(axis=-1)(drop1)

    # Deconvolution, 6 x 8 x 512 -> 14 x 18 x 256
    conv2 = layers.Conv2DTranspose(config["complexity_2"] * config["g_layers"][2], (3, 4), 1, use_bias=False)(bn1)
    lrelu2 = layers.LeakyReLU(config.lrelu)(conv2)
    drop2 = layers.Dropout(config.dropout)(lrelu2)
    bn2 = layers.BatchNormalization(axis=-1)(drop2)

    # Output layer, 17 x 21 x 128 -> 20 x 24 x nchannels
    score = layers.Conv2DTranspose(nchannels, (4, 6), 2)(bn2)
    o = score if config.gumbel else tf.keras.activations.sigmoid(score) # NOTE: check
    return tf.keras.Model(z, o, name="generator")


def define_critic(config, nchannels=2):
    """
    >>> critic = define_critic()
    """
    x = tf.keras.Input(shape=(20, 24, nchannels + 1))

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

wgan.generator = define_generator(wandb.config, nchannels=2)
wgan.critic = define_critic(wandb.config, nchannels=2)
# %% generator
from keras import ops
u, z = next(iter(train))
batch_size = u.shape[0]
random_latent_vectors = tf.random.normal((batch_size, 100))
random_vector_labels = ops.concatenate([random_latent_vectors, z[:, None]], axis=1)
fake_u = wgan.generator(random_vector_labels)

# %% discriminator
z = ops.repeat(z[..., None, None], repeats=[u.shape[1] * u.shape[2]])
z = ops.reshape(z, (-1, u.shape[1], u.shape[2], 1))
u = ops.concatenate([u, z], axis=-1)

# %%
wgan.critic(u)
# %% ref critic function
