import tensorflow as tf
from tensorflow.keras import layers


def BatchNormalization3D(x, axis=-1):
    """NOTE: Assumes channels last. For use with 4D (temporal) data.
    
    Batch normalisation is on a per-channel basis so this just reshapes
    so that we can use the usual keras BatchNormalization."""
    n, t, h, w, c =  tf.keras.backend.int_shape(x)
    x = layers.Reshape((t * h * w, c))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Reshape((t, h, w, c))(x)
    return x

