import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D as Downsampling

from .upsample import Conv2DTransposeNN as Upsampling

def ResidualUpBlock(input, filters, kernel_size, config):
    residual = input
    residual = Upsampling(kernel_size)(residual)

    output = layers.Conv2DTranspose(filters, kernel_size, 1, use_bias=False)(input)
    output = layers.LeakyReLU(config['lrelu'])(output)
    output = layers.Dropout(config['dropout'])(output)
    if config['normalize_generator']:
        output = layers.BatchNormalization(axis=-1)(output)  # normalise along features layer (1024)
    else:
        output = output

    return output + residual


def ResidualDownBlock(input, filters, kernel_size:tuple, strides:tuple, config):
    residual = input
    residual = Downsampling(kernel_size, strides)(residual)
    
    output = layers.Conv2D(filters, kernel_size, strides,
                          kernel_initializer=tf.keras.initializers.GlorotUniform())(input)
    output = layers.LeakyReLU(config['lrelu'])(output)
    output = layers.Dropout(config['dropout'])(output)

    return output + residual