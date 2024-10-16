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
  def __init__(self, kernel_size:tuple) -> tf.Tensor:
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

  def build(self, input_shape):
    # first conv2dtranspose -- zero-insertion
    filters = input_shape[-1]

    w = np.zeros(self.kernel_size, dtype=np.float32)
    w[0, 0] = 1
    w = w[..., np.newaxis, np.newaxis] # HWCB
    w = np.repeat(w, filters, axis=2)
    w = np.repeat(w, filters, axis=3)
    b = np.array([0.], dtype=np.float32)
    b = np.repeat(b, filters, axis=0)
    w = tf.Variable(tf.constant(w))
    b = tf.Variable(b)

    layer0 = layers.Conv2DTranspose(
      filters,
      self.kernel_size,
      strides=self.kernel_size,
      padding='valid'
      )
    layer0.build(input_shape)
    layer0.set_weights([w, b])
    self.layer0 = layer0

    # second conv2d -- fixed kernel
    ones = np.ones(self.kernel_size, dtype=np.float32)
    w = np.zeros(self.kernel2_size, dtype=np.float32)
    w[:-1, :-1] = ones
    w = w[..., np.newaxis, np.newaxis]
    w = np.repeat(w, filters, axis=2)
    w = np.repeat(w, filters, axis=3)
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
    output = output[:, :h, :w, :] # __get_item__()
    return output


# %% tests (add to pytest script later)

# import pytest
# @pytest.fixture
def conv2d_transpose_nn():
    return Conv2DTransposeNN(kernel_size=(2, 2))

def test_initialization(conv2d_transpose_nn):
    assert conv2d_transpose_nn.kernel_size == (2, 2)
    assert conv2d_transpose_nn.kernel2_size == (3, 3)

def test_build_method(conv2d_transpose_nn):
    input_shape = (1, 4, 4, 3)
    conv2d_transpose_nn.build(input_shape)
    assert hasattr(conv2d_transpose_nn, 'layer0')
    assert hasattr(conv2d_transpose_nn, 'layer1')

def test_output_shape():
    layer = Conv2DTransposeNN(kernel_size=(2, 2))
    input_tensor = tf.random.normal((1, 4, 4, 3))
    output = layer(input_tensor)
    assert output.shape == (1, 8, 8, 3)

def test_output_shape_different_kernel():
    layer = Conv2DTransposeNN(kernel_size=(3, 3))
    input_tensor = tf.random.normal((1, 5, 5, 2))
    output = layer(input_tensor)
    assert output.shape == (1, 15, 15, 2)

def test_output_values():
    layer = Conv2DTransposeNN(kernel_size=(2, 2))
    input_tensor = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    expected_output = tf.constant([
        [
            [[1.0], [1.0], [2.0], [2.0]],
            [[1.0], [1.0], [2.0], [2.0]],
            [[3.0], [3.0], [4.0], [4.0]],
            [[3.0], [3.0], [4.0], [4.0]]
        ]
    ])
    output = layer(input_tensor)
    tf.debugging.assert_near(output, expected_output, atol=1e-5)

def test_multiple_channels():
    layer = Conv2DTransposeNN(kernel_size=(2, 2))
    input_tensor = tf.random.normal((1, 4, 4, 3))
    output = layer(input_tensor)
    assert output.shape == (1, 8, 8, 3)

def test_batch_processing():
    layer = Conv2DTransposeNN(kernel_size=(2, 2))
    input_tensor = tf.random.normal((5, 4, 4, 3))
    output = layer(input_tensor)
    assert output.shape == (5, 8, 8, 3)

def test_large_kernel():
    layer = Conv2DTransposeNN(kernel_size=(10, 10))
    input_tensor = tf.random.normal((1, 3, 3, 20))
    output = layer(input_tensor)
    assert output.shape == (1, 30, 30, 20)

def test_non_square_kernel():
    layer = Conv2DTransposeNN(kernel_size=(2, 3))
    input_tensor = tf.random.normal((1, 4, 4, 3))
    output = layer(input_tensor)
    assert output.shape == (1, 8, 12, 3)

def test_invalid_input_shape():
    layer = Conv2DTransposeNN(kernel_size=(2, 2))
    with pytest.raises(ValueError):
        layer(tf.random.normal((1, 4, 4)))  # Missing channel dimension

def test_model_integration():
    inputs = tf.keras.Input(shape=(4, 4, 3))
    x = Conv2DTransposeNN(kernel_size=(2, 2))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    assert model.output_shape == (None, 8, 8, 3)


if __name__ == "__main__":
  conv2d_transpose_nn = conv2d_transpose_nn()
  test_initialization(conv2d_transpose_nn)
  test_build_method(conv2d_transpose_nn)
  test_output_shape()
  test_output_shape_different_kernel()
  test_output_values()
  test_multiple_channels()
  test_batch_processing()
  test_large_kernel()
  test_non_square_kernel()
  test_invalid_input_shape()
  test_model_integration()

  # run example script
  fig, axs = plt.subplots(1, 2)
  x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
  x = x[np.newaxis, ..., np.newaxis]
  x = np.repeat(x, 20, axis=-1);print(x.shape) # BHWC?
  x = tf.constant(x)

  y = Conv2DTransposeNN((10, 10))(x)

  print(type(y))
  x = x[0, ..., 1]
  y = y[0, ..., 1]
  axs[0].imshow(x)
  axs[1].imshow(y)

  print(x.shape)
  print(y.shape)

# %% END
