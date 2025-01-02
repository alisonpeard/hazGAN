"""Helper functions for running evtGAN in TensorFlow."""
import os
import numpy as np
import xarray as xr
import tensorflow as tf

__all__ = ['unpad', 'gaussian_blur']

def unpad(tensor, paddings=None):
    """Mine: remove Tensor paddings"""
    if paddings is None:
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    tensor = tf.convert_to_tensor(tensor)  # incase its a np.array
    unpaddings = [
        (
            slice(pad.numpy()[0], -pad.numpy()[1])
            if sum(pad.numpy() > 0)
            else slice(None, None)
        )
        for pad in paddings
    ]
    return tensor[unpaddings]



def gaussian_blur(img, kernel_size=11, sigma=5):
    """See: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319"""

    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    img = tf.cast(img, tf.float32)
    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(
        img, gaussian_kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
    )

