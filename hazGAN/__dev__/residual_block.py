# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains common building blocks for neural networks."""
# %%

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Text

# Import libraries
import tensorflow as tf
import functools
import six

# from official.modeling import tf_utils # clone_initializer, get_activation
from official.vision.modeling.layers import nn_layers


def clone_initializer(initializer):
  # Keras initializer is going to be stateless, which mean reusing the same
  # initializer will produce same init value when the shapes are the same.
  if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
  # When the input is string/dict or other serialized configs, caller will
  # create a new keras Initializer instance based on that, and we don't need to
  # do anything
  return initializer


def get_activation(identifier, use_keras_layer=False, **kwargs):
  """Maps an identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Prefers using keras layers when use_keras_layer=True. Now it only supports
  'relu', 'linear', 'identity', 'swish', 'mish', 'leaky_relu', and 'gelu'.

  Args:
    identifier: String name of the activation function or callable.
    use_keras_layer: If True, use keras layer if identifier is allow-listed.
    **kwargs: Keyword arguments to use to instantiate an activation function.
      Available only for 'leaky_relu' and 'gelu' when using keras layers.
      For example: get_activation('leaky_relu', use_keras_layer=True, alpha=0.1)

  Returns:
    A Python function corresponding to the activation function or a keras
    activation layer when use_keras_layer=True.
  """
  if isinstance(identifier, six.string_types):
    identifier = str(identifier).lower()
    if use_keras_layer:
      keras_layer_allowlist = {
          "relu": "relu",
          "linear": "linear",
          "identity": "linear",
          "swish": "swish",
          "sigmoid": "sigmoid",
          "relu6": tf.nn.relu6,
          "leaky_relu": functools.partial(tf.nn.leaky_relu, **kwargs),
          "hard_swish": activations.hard_swish,
          "hard_sigmoid": activations.hard_sigmoid,
          "mish": activations.mish,
          "gelu": functools.partial(tf.nn.gelu, **kwargs),
      }
      if identifier in keras_layer_allowlist:
        return tf.keras.layers.Activation(keras_layer_allowlist[identifier])
    name_to_fn = {
        "gelu": activations.gelu,
        "simple_swish": activations.simple_swish,
        "hard_swish": activations.hard_swish,
        "relu6": activations.relu6,
        "hard_sigmoid": activations.hard_sigmoid,
        "identity": activations.identity,
        "mish": activations.mish,
    }
    if identifier in name_to_fn:
      return tf.keras.activations.get(name_to_fn[identifier])
  return tf.keras.activations.get(identifier)


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResidualBlock(tf.keras.layers.Layer):
  """A residual block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               resnetd_shortcut=False,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_explicit_padding: bool = False,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               bn_trainable=True,
               **kwargs):
    """Initializes a residual block with BN after convolutions.

    Args:
      filters: An `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
      strides: An `int` block stride. If greater than 1, this block will
        ultimately downsample the input.
      use_projection: A `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      resnetd_shortcut: A `bool` if True, apply the resnetd style modification
        to the shortcut connection. Not implemented in residual blocks.
      stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      bn_trainable: A `bool` that indicates whether batch norm layers should be
        trainable. Default to True.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(ResidualBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
    self._use_explicit_padding = use_explicit_padding
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._norm = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)
    self._bn_trainable = bn_trainable

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf.keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable,
          synchronized=self._use_sync_bn,
      )

    conv1_padding = 'same'
    # explicit padding here is added for centernet
    if self._use_explicit_padding:
      self._pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
      conv1_padding = 'valid'

    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding=conv1_padding,
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )

    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable,
        synchronized=self._use_sync_bn,
    )

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters,
          out_filters=self._filters,
          se_ratio=self._se_ratio,
          kernel_initializer=tf_utils.clone_initializer(
              self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    super(ResidualBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'resnetd_shortcut': self._resnetd_shortcut,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_explicit_padding': self._use_explicit_padding,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'bn_trainable': self._bn_trainable
    }
    base_config = super(ResidualBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    if self._use_explicit_padding:
      inputs = self._pad(inputs)
    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)

  # %%