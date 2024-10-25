# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %%
class Conv2DTransposeNN(tf.keras.layers.Layer):
  """Better upsampling to create any shape to match Conv2DTranspose.

  Refs:
  .. [1] https://arxiv.org/abs/1907.06515
  """
  def __init__(self, filters:int, kernel_size:tuple) -> tf.Tensor:
    """
    Nearest neighbour interpolation layer.

    Upsample (increase resolution) of a tensor by nearest neighbour sampling.
    This uses two transpose convolution layers to upsample a tensor. The output
    size is determined by the kernel sizes analogously to the
    tensorflow.keras.layers.Conv2DTranspose function.

    Args:
      kernel_size: tuple, specifying the size of the
        transposed convolution window.
      filters

    """
    super(Conv2DTransposeNN, self).__init__()
    
    self.kernel_size = kernel_size
    self.kernel2_size = (kernel_size[0]+1, kernel_size[1]+1)
    self.filters = filters

  def build(self, input_shape):
    # first conv2dtranspose -- zero-insertion
    filters = input_shape[-1]

    # (h, w, out, in)
    w = np.zeros((*self.kernel_size, filters, filters), dtype=np.float32)
    w[0, 0, :, :] = np.eye(filters)
    b = np.array([0.], dtype=np.float32)
    b = np.repeat(b, filters, axis=0)
    w = tf.Variable(tf.constant(w))
    b = tf.Variable(b)

    layer0 = layers.Conv2DTranspose(
      filters,
      self.kernel_size,
      strides=self.kernel_size,
      padding='same'
      )
    layer0.build(input_shape)
    layer0.set_weights([w, b])
    self.layer0 = layer0

    # second conv2d -- fixed kernel
    w = np.zeros((*self.kernel2_size, filters, filters), dtype=np.float32)
    ones = np.ones(self.kernel_size, dtype=np.float32)
    w[:-1,:-1, ...] = np.einsum('ab,cd->abcd', ones, np.eye(filters))
    b = np.array([0.], dtype=np.float32)
    b = np.repeat(b, filters, axis=0)
    w = tf.Variable(tf.constant(w))
    b = tf.Variable(b)

    layer1 = layers.Conv2DTranspose(
      filters,
      self.kernel2_size,
      padding='valid',
      strides=1
      )
    layer1.build(input_shape)
    layer1.set_weights([w, b])
    self.layer1 = layer1


  def call(self, input):
    output = self.layer0(input)
    h = tf.shape(output)[1]
    w = tf.shape(output)[2]
    output = self.layer1(output)
    filters = self.filters
    output = output[:, :h, :w, :filters] # __get_item__()
    return output


# %% tests (add to pytest script later)
if __name__ == "__main__":
  # x = np.random.normal(size=(1, 5, 5, 1))
  nchan = 2
  x = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
  x = x.reshape(1, 3, 3, 1)
  x = np.repeat(x, nchan, -1)
  x = tf.constant(x)

  kernel_size = (2, 2)
  y = Conv2DTransposeNN(nchan, kernel_size)(x)
  y2 = layers.Conv2DTranspose(nchan, kernel_size, 1, padding='valid', use_bias=False)(x)

  print('Initial shape:', x.shape)
  print('Shape after upsampling:', y.shape)
  print("Shape after deconvolution:", y2.shape)

  channel = 1
  fig, axs = plt.subplots(1, 3)
  axs[0].imshow(x[0, ..., channel], vmin=0, vmax=10)
  axs[1].imshow(y[0, ..., channel], vmin=0, vmax=10)
  axs[2].imshow(y2[0, ..., channel])

  assert y.shape == y2.shape

# %% DEV BELOW HERE
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1) # stacked along H-dimension
    output = tf.transpose(output, [0,2,3,1])                     # NWCH
    output = tf.nn.depth_to_space(output, block_size=2)          # convert to blocks of size 2 x 2
    output = tf.transpose(output, [0,3,1,2])                     # NHWC
    output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output
# %%
help(tf.depth_to_space)
# %%
