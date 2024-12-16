"""Wrappers for all the layers used in the GAN."""
# %%
from keras import layers
from keras import initializers
from keras.layers import Layer
import tensorflow as tf


ortho_initializer = initializers.Orthogonal
henorm_initializer = initializers.HeNormal # best for LeakyReLU and matches data
normal_initializer = initializers.RandomNormal # mean=0.0, stddev=0.02
uniform_initializer = initializers.RandomUniform


class Embedding(layers.Embedding):
    """Wrapper for Embedding with orthogonal initialization."""
    def __init__(self, nlabels, output_dim, initializer=uniform_initializer, *args, **kwargs):
        scale = 1.0 / (output_dim ** 0.5)
        initializer = initializer(minval=-scale, maxval=scale)
        super(Embedding, self).__init__(nlabels, output_dim, *args, **kwargs)


class Dense(layers.Dense):
    """Wrapper for Dense with orthogonal initialization."""
    def __init__(self, filters, initializer=henorm_initializer, *args, **kwargs):
        super(Dense, self).__init__(filters, kernel_initializer=initializer(), *args, **kwargs)


class Conv2D(layers.Conv2D):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, padding, initializer=henorm_initializer, **kwargs):
        super(Conv2D, self).__init__(filters, kernel_size, stride, padding,
                                          kernel_initializer=initializer(),
                                          **kwargs)


class Conv2DTranspose(layers.Conv2DTranspose):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, initializer=henorm_initializer, **kwargs):
        super(Conv2DTranspose, self).__init__(filters, kernel_size, stride,
                                          kernel_initializer=initializer(),
                                          **kwargs)
        

class BatchNormalization(layers.BatchNormalization):
    """Wrapper for BatchNormalization with orthogonal initialization."""
    def __init__(self, initializer=henorm_initializer, *args, **kwargs):
        super(BatchNormalization, self).__init__(gamma_initializer=initializer(),
                                                 *args, **kwargs)
        

class GumbelEsque(Layer):
  def __init__(self, epsilon=1e-8, *args, **kwargs):
    super(GumbelEsque, self).__init__(*args, **kwargs)
    self.epsilon = epsilon

  def call(self, inputs):
    uniform = tf.math.sigmoid(inputs)
    out = -tf.math.log(uniform)
    gumbel = -tf.math.log(out)
    return gumbel

    uniform = tf.clip_by_value(
        tf.math.sigmoid(inputs),
        clip_value_min = self.epsilon,
        clip_value_max = 1. - self.epsilon
    )
    return -tf.math.log(-tf.math.log(uniform))
# %%