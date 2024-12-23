"""Wrappers for all the layers used in the GAN."""
# %%
from keras import layers
from keras import initializers


ortho_initializer = initializers.Orthogonal
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
    def __init__(self, filters, initializer=ortho_initializer, *args, **kwargs):
        super(Dense, self).__init__(filters, kernel_initializer=initializer(), *args, **kwargs)


class Conv2D(layers.Conv2D):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, padding, initializer=ortho_initializer, **kwargs):
        super(Conv2D, self).__init__(filters, kernel_size, stride, padding,
                                          kernel_initializer=initializer(),
                                          **kwargs)


class Conv2DTranspose(layers.Conv2DTranspose):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, initializer=ortho_initializer, **kwargs):
        super(Conv2DTranspose, self).__init__(filters, kernel_size, stride,
                                          kernel_initializer=initializer(),
                                          **kwargs)
        

class BatchNormalization(layers.BatchNormalization):
    """Wrapper for BatchNormalization with orthogonal initialization."""
    def __init__(self, initializer=normal_initializer, *args, **kwargs):
        super(BatchNormalization, self).__init__(gamma_initializer=initializer(),
                                                 *args, **kwargs)
# %%