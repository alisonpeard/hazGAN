"""Helper functions for running evtGAN in TensorFlow."""
import os
import numpy as np
import xarray as xr
import torch

__all__ = ['unpad']


def unpad(tensor, paddings=(1, 1, 1, 1)):
    """Mine: remove Tensor paddings"""
    left, right, top, bottom = paddings
    return tensor[:, :, top:-bottom, left:-right]


def gaussian_blur(img, kernel_size=11, sigma=5):
    """See: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319"""
    raise NotImplementedError("No PyTorch implementation yet.")
    # def gauss_kernel(channels, kernel_size, sigma):
    #     ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    #     xx, yy = tf.meshgrid(ax, ax)
    #     kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    #     kernel = kernel / tf.reduce_sum(kernel)
    #     kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
    #     return kernel

    # img = tf.cast(img, tf.float32)
    # gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    # gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    # return tf.nn.depthwise_conv2d(
    #     img, gaussian_kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
    # )

