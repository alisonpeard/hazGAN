#%%
import functools
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import wrappers
else:
    from . import wrappers

# %%
class FractionalUpsample(layers.Layer):
    """Upsample by a fractional factor.
    
    To upsample by a factor of L/M, we first upsample by L and then downsample by M.
    """
    def __init__(self, filters, kernel_size=2, strides=1,
                 padding="valid", testing=False, **kwargs) -> None:
        super(FractionalUpsample, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.testing = testing
        self.upsample = layers.UpSampling2D
        self.downsample = layers.AveragePooling2D

    def calculate_output_shape(self, input_shape) -> tuple[int,int,int,int]:
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
        self.conv1x1 = wrappers.Conv2D(self.filters, 1, padding='same')

    def call(self, input) -> tf.Tensor:
        upsampled = self.upsample(size=self.upsampling_factor)(input)
        resampled = self.downsample(
            pool_size=self.downsampling_factor,
            strides=self.downsampling_factor,
            padding="valid")(upsampled)
        if self.testing:
            output = resampled
        else:
            output = self.conv1x1(resampled)
        return output
    

class FractionalDownsample(layers.Layer):
    """Downsample by a non-integer factor.
    
    https://www.youtube.com/watch?app=desktop&v=GWt6Fu05voI&t=0s # 00:15:11
    """
    def __init__(self, filters, kernel_size, strides=1,
                 padding="valid", testing=False, **kwargs) -> None:
        super(FractionalDownsample, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.testing = testing
        self.downsample = layers.AveragePooling2D

    def calculate_output_shape(self, input_shape) -> tuple[int,int,int,int]:
        """Calculate the output shape of a Conv2DTranspose layer."""
        self.input_filters = input_shape[-1]
        height, width = input_shape[1:3]
        stride_height, stride_width = self.strides
        if self.padding == 'valid':
            output_height = (height - self.kernel_size[0]) // stride_height + 1
            output_width = (width - self.kernel_size[1]) // stride_width + 1
        elif self.padding == 'same':
            output_height = height // stride_height
            output_width = width // stride_width
        else:
            raise ValueError(f"Invalid padding: {self.padding}")
        self.output_shape = (input_shape[0], output_height, output_width, self.filters)

    def build(self, input_shape) -> None:
        self.calculate_output_shape(input_shape)
        self.conv1x1 = wrappers.Conv2D(self.filters, 1, padding="same")

    def call(self, input) -> tf.Tensor:
        """Note this uses 1d convolutions to increase the depth."""
        downsampled = self.downsample(
            pool_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding
            )(input)
        if self.testing:
            output = downsampled
        else:
            output = self.conv1x1(downsampled)
        return output
    

class UpResidual(layers.Layer):
    """Residual block with Conv2DTranspose and regularisation layers.

    https://www.youtube.com/watch?app=desktop&v=GWt6Fu05voI&t=0s # 00:15:11
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 use_bias=True, lrelu=False, dropout=None, **kwargs) -> None:
        super(UpResidual, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.lrelu = lrelu
        self.dropout = dropout
        self.use_bias = use_bias

    def build(self, input_shape) -> None:
        self.upsample = FractionalUpsample(self.filters, self.kernel_size, self.strides, self.padding)
        self.deconv = wrappers.Conv2DTranspose(self.filters, self.kernel_size, self.strides,
                                               self.padding, use_bias=self.use_bias)
        self.relu = layers.LeakyReLU(self.lrelu) if self.lrelu is not None else layers.Identity()
        self.dropout = layers.Dropout(self.dropout) if self.dropout else layers.Identity()
        self.normalise = wrappers.BatchNormalization(axis=-1)

    def call(self, input) -> tf.Tensor:
        shortcut = self.upsample(input)
        x = self.deconv(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.normalise(x)
        return x + shortcut
    

class DownResidual(layers.Layer):
    """Residual block with two Conv2D layers."""
    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 use_bias=True, lrelu=False, dropout=None, **kwargs) -> None:
        super(DownResidual, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.lrelu = lrelu
        self.dropout = dropout

    def build(self, input_shape) -> None:
        self.downsample = FractionalDownsample(self.filters, self.kernel_size,
                                               self.strides, self.padding)
        self.conv = wrappers.Conv2D(self.filters, self.kernel_size, self.strides,
                                    self.padding, use_bias=self.use_bias)
        self.relu = layers.LeakyReLU(self.lrelu) if self.lrelu else layers.Identity()
        self.dropout = layers.Dropout(self.dropout) if self.dropout else layers.Identity()
        self.normalise = layers.LayerNormalization(axis=-1)


    def call(self, input) -> tf.Tensor:
        shortcut = self.downsample(input)
        x = self.conv(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.normalise(x)
        return x + shortcut

# %%
if __name__ == "__main__":
    # Sanity checks and visualisation (turn to tests later)
    import matplotlib.pyplot as plt

    MATCH_FIRST = True # only set True for testing

    if True:
        # compare to Conv2DTranspose
        print("\nUpsampling:\n----------")
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
        output = FractionalUpsample(512, 3, 1, testing=MATCH_FIRST)(input)
        print(output.shape)

        fig, axs = plt.subplots(1, 2)
        vmin = np.min(input[0, ..., 0])
        vmax = np.max(input[0, ..., 0])
        im = axs[0].imshow(input[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(output[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[1])
        fig.suptitle("ResidualUpsample first")


        output = FractionalUpsample(256, (3, 4), 1, testing=MATCH_FIRST)(output)
        print(output.shape)

        fig, axs = plt.subplots(1, 2)
        vmin = np.min(input[0, ..., 0])
        vmax = np.max(input[0, ..., 0])
        im = axs[0].imshow(input[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(output[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[1])
        fig.suptitle("ResidualUpsample final")

    # test ResidualDownsample
    if True:
        print("\nDownsampling:\n----------")
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
        output = FractionalDownsample(64, (4,5), 2, testing=MATCH_FIRST)(input)
        print(output.shape)

        fig, axs = plt.subplots(1, 2)
        vmin = np.min(input[0, ..., 0])
        vmax = np.max(input[0, ..., 0])
        im = axs[0].imshow(input[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(output[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[1])
        fig.suptitle("ResidualDownsample first")

        output = FractionalDownsample(128, (3,4), testing=MATCH_FIRST)(output)
        print(output.shape)
        output = FractionalDownsample(256, (3, 3), testing=MATCH_FIRST)(output)
        print(output.shape)

        fig, axs = plt.subplots(1, 2)
        vmin = np.min(input[0, ..., 0])
        vmax = np.max(input[0, ..., 0])
        im = axs[0].imshow(input[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(output[0, ..., 0], vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axs[1])
        fig.suptitle("ResidualDownsample final")

        # %%
        up_block = UpResidual(256, (3, 3), 1, padding='same', lrelu=0.2, dropout=0.5)
        down_block = DownResidual(256, (3, 3), 1, padding='same', lrelu=0.2, dropout=0.5)
        # %%

# %%
