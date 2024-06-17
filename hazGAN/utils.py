"""Helper functions for running evtGAN in TensorFlow."""

import numpy as np
import tensorflow as tf

SEED = 42

# bounds (EPSG:4326)
xmin = 80.0
xmax = 95.0
ymin = 10.0
ymax = 25.0


def diff(x, d=1):
    """Difference a (time series) array."""
    return x[d:] - x[:-d]


def translate_indices(i, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = np.argwhere(indices.reshape(dims[0], dims[1]) == i)
    return tuple(*map(tuple, x))


def translate_indices_r(i, j, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = indices.reshape(dims[0], dims[1])[i, j]
    return x


def unpad(tensor, paddings=tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])):
    """Mine: remove Tensor paddings"""
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


def sliding_windows(x, size, step=1):
    n, *_ = x.shape
    window_indices = sliding_window_indices(size, n, step)
    return x[window_indices, ...]


def sliding_window_indices(size, n, step=1):
    windows = []
    i = 0
    for i in range(0, n - size, step):
        windows.append(np.arange(i, i + size, 1))
    return np.array(windows)


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


