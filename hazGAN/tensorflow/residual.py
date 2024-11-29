#%%
import functools
import tensorflow as tf
from tensorflow.keras import layers
# from . import wrappers

import numpy as np
import wrappers
import matplotlib.pyplot as plt

# %%
class ResidualUpsample(layers.Layer):
    """Upsample by a non-integer factor.
    
    https://www.youtube.com/watch?app=desktop&v=GWt6Fu05voI&t=0s # 00:15:11
    """
    def __init__(self, filters, kernel_size=2, strides=1,
                 padding="valid", **kwargs) -> None:
        super(ResidualUpsample, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding


    def calculate_output_shape(self, input_shape) -> tuple:
        """Calculate the output shape of a Conv2DTranspose layer."""
        self.input_filters = input_shape[-1]
        height, width = input_shape[1:3]
        stride_height, stride_width = self.strides
        if self.padding == 'valid':
            output_height = (height - 1) * stride_height + self.kernel_size[0]
            output_width = (width - 1) * stride_width + self.kernel_size[1]
        elif self.padding == 'same':
            output_height = height * stride_height
            output_width = width * stride_width
        else:
            raise ValueError(f"Invalid padding: {self.padding}")
        self.upsampling_factor = (output_height, output_width)
        self.downsampling_factor = (height, width)
        self.output_shape = (input_shape[0], output_height, output_width, self.filters)


    def build(self, input_shape) -> None:
            self.calculate_output_shape(input_shape)


    def call(self, input) -> tf.Tensor:
        """Note this uses 1d convolutions to increase the depth."""
        upsampled = layers.UpSampling2D(size=self.upsampling_factor)(input)
        resampled = layers.AveragePooling2D(pool_size=self.downsampling_factor)(upsampled)
        output = layers.Conv2D(self.filters, 1, padding='same')(resampled)
        return output
    


class ResidualDownsample(layers.Layer):
    """Downsample by a non-integer factor.
    
    https://www.youtube.com/watch?app=desktop&v=GWt6Fu05voI&t=0s # 00:15:11
    """
    def __init__(self, filters, kernel_size, strides=1,
                 padding="valid", **kwargs) -> None:
        super(ResidualDownsample, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding


    def call(self, input) -> tf.Tensor:
        """Note this uses 1d convolutions to increase the depth."""
        downsampled = layers.AveragePooling2D(pool_size=self.kernel_size,
                                              strides=self.strides,
                                              padding=self.padding)(input)
        output = layers.Conv2D(self.filters, 1, padding="same")(downsampled)
        return output
    

class ResidualUpBlock(layers.Layer):
    """Residual block with two Conv2D layers."""
    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 dropout=0.2, **kwargs) -> None:
        super(ResidualUpBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.upsample = ResidualUpsample(self.filters, self.kernel_size, self.strides, self.padding)
        self.deconv = layers.Conv2DTranspose(self.filters, self.kernel_size, self.strides, self.padding)
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout)
        self.normalise = layers.BatchNormalization() # TODO: (I have custom normalisation)

    def build(self, input_shape) -> None:
        # TODO: check what this should be
        super(ResidualUpBlock, self).build(input_shape)
        self.upsample.build(input_shape)
        self.deconv.build(input_shape)

    def call(self, input) -> tf.Tensor:
        shortcut = self.upsample(input)
        x = self.deconv(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.normalise(x)
        return x + shortcut
    

class ResidualDownBlock(layers.Layer):
    """Residual block with two Conv2D layers."""
    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 dropout=0.2, **kwargs) -> None:
        super(ResidualDownBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.downsample = ResidualDownsample(self.filters, self.kernel_size, self.strides, self.padding)
        self.conv = layers.Conv2D(self.filters, self.kernel_size, self.strides, self.padding)
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout)
        self.normalise = layers.BatchNormalization() # TODO: (I have custom normalisation)

    def build(self, input_shape) -> None:
        super(ResidualDownBlock, self).build(input_shape)
        self.downsample.build(input_shape)
        self.conv.build(input_shape)

    def call(self, input) -> tf.Tensor:
        shortcut = self.downsample(input)
        x = self.conv(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.normalise(x)
        return x + shortcut

# %%

if __name__ == "__main__":
    # test ResidualUpsample
    if True:
        input = np.random.normal(size=(1, 5, 5, 1024))
        input = tf.constant(input, dtype=tf.float32)

        upsample_layer = ResidualUpsample(512, (3, 3))
        output = upsample_layer(input)
        assert (1, 7, 7, 512) == output.shape, f"Expected {(1, 7, 7, 512)}, got {output.shape}"

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        im = axs[0].imshow(input[0, ..., 0])
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(output[0, ..., 0])
        plt.colorbar(im, ax=axs[1])

        # compare to Conv2DTranspose
        input = np.random.normal(size=(1, 5, 5, 1024))
        input = tf.constant(input, dtype=tf.float32)
        print(input.shape)
        output = layers.Conv2DTranspose(512, 3, 1)(input)
        print(output.shape)
        output = layers.Conv2DTranspose(256, (3, 4), 1)(output)
        print(output.shape)

        # test ResidualUpsample
        input = np.random.normal(size=(1, 5, 5, 1024))
        input = tf.constant(input, dtype=tf.float32)
        print(input.shape)
        output = ResidualUpsample(512, 3, 1)(input)
        print(output.shape)
        output = ResidualUpsample(256, (3, 4), 1)(output)
        print(output.shape)

    # test ResidualDownsample
    if True:
        # compare to Conv2D
        input = np.random.normal(size=(1, 20, 24, 2))
        print(input.shape)
        output = layers.Conv2D(64, (4,5), 2)(input)
        print(output.shape)
        output = layers.Conv2D(128, (3,4))(output)
        print(output.shape)
        output = layers.Conv2D(256, (3, 3))(output)
        print(output.shape)
        
        # test ResidualDownsample
        input = np.random.normal(size=(1, 20, 24, 2))
        print(input.shape)
        output = ResidualDownsample(64, (4,5), 2)(input)
        print(output.shape)
        output = ResidualDownsample(128, (3,4))(output)
        print(output.shape)
        output = ResidualDownsample(256, (3, 3))(output)
        print(output.shape)