"""Wrappers for all the layers used in the GAN."""
# %%
from keras import layers
from keras import initializers
from keras.layers import Layer
import tensorflow as tf


ortho_initializer = initializers.Orthogonal
henorm_initializer = initializers.HeNormal # best for LeakyReLU and matches data
normal_initializer = initializers.RandomNormal # mean=0.0, stddev=0.02
uniform_initializer = initializers.RandomUniform


class Embedding(layers.Embedding):
    """Wrapper for Embedding with orthogonal initialization."""
    def __init__(self, nlabels, output_dim, initializer=uniform_initializer, *args, **kwargs):
        scale = 1.0 / (output_dim ** 0.5)
        initializer = initializer(minval=-scale, maxval=scale)
        super(Embedding, self).__init__(nlabels, output_dim, *args, **kwargs)


class Dense(layers.Dense):
    """Wrapper for Dense with orthogonal initialization."""
    def __init__(self, filters, initializer=henorm_initializer, *args, **kwargs):
        super(Dense, self).__init__(filters, kernel_initializer=initializer(), *args, **kwargs)


class Conv2D(layers.Conv2D):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, padding, initializer=henorm_initializer, **kwargs):
        super(Conv2D, self).__init__(filters, kernel_size, stride, padding,
                                          kernel_initializer=initializer(),
                                          **kwargs)


class Conv2DTranspose(layers.Conv2DTranspose):
    """Wrapper for Conv2D with orthogonal initialization."""
    def __init__(self, filters, kernel_size, stride, initializer=henorm_initializer, **kwargs):
        super(Conv2DTranspose, self).__init__(filters, kernel_size, stride,
                                          kernel_initializer=initializer(),
                                          **kwargs)
        

class BatchNormalization(layers.BatchNormalization):
    """Wrapper for BatchNormalization with orthogonal initialization."""
    def __init__(self, initializer=henorm_initializer, *args, **kwargs):
        super(BatchNormalization, self).__init__(gamma_initializer=initializer(),
                                                 *args, **kwargs)
        

class GumbelEsque(Layer):
  def __init__(self, epsilon=1e-6, *args, **kwargs):
    super(GumbelEsque, self).__init__(*args, **kwargs)
    self.epsilon = epsilon

  def call(self, inputs):
    if False:
        uniform = tf.math.sigmoid(inputs)
        out = -tf.math.log1p(uniform)
        gumbel = -tf.math.log1p(out)
    else:
        uniform = tf.clip_by_value(
            tf.math.sigmoid(inputs),
            clip_value_min = self.epsilon,
            clip_value_max = 1. - self.epsilon
        )
        gumbel = -tf.math.log(-tf.math.log(uniform))
    return gumbel
# %% CLAUDE
import tensorflow as tf

def differentiable_batch_uniform_transform(x, smoothing_factor=1e-3):
    """
    Differentiable transformation to uniform distribution across batch dimension.
    
    Parameters:
    -----------
    x : tf.Tensor
        Input tensor of shape [batch_size, ...]
    smoothing_factor : float, optional
        Controls the smoothness of the ranking approximation
    
    Returns:
    --------
    tf.Tensor
        Transformed tensor with uniform-like distribution
    """
    # Reshape to separate batch and feature dimensions
    original_shape = tf.shape(x)
    x_flat = tf.reshape(x, [original_shape[0], -1])
    
    def soft_rank(feature_column):
        comparisons = tf.sigmoid( # broadcasted comparisons in 2D
            (
            feature_column[:, tf.newaxis] - feature_column[tf.newaxis, :]
            ) / smoothing_factor
            )
        soft_ranks = tf.reduce_sum(comparisons, axis=1)
        numerator = soft_ranks - tf.reduce_min(soft_ranks)
        denominator = tf.reduce_max(soft_ranks) - tf.reduce_min(soft_ranks)
        normalized_ranks = numerator / denominator
        return normalized_ranks
    
    uniform_distribution = tf.transpose(
        tf.map_fn(soft_rank, tf.transpose(x_flat))
    )
    
    return tf.reshape(uniform_distribution, original_shape)

# Demonstration function
def demonstrate_differentiable_transform():
    # Create a sample tensor
    x = tf.Variable([
        [1.0, 5.0],
        [3.0, 2.0],
        [2.0, 4.0],
        [4.0, 3.0]
    ])
    
    # Demonstrate differentiability with gradient computation
    with tf.GradientTape() as tape:
        transformed = differentiable_batch_uniform_transform(x)
        # Example loss (just to show gradient computation)
        loss = tf.reduce_mean(transformed)
    
    # Compute gradients
    grads = tape.gradient(loss, [x])
    
    print("Original tensor:")
    print(x)
    print("\nTransformed tensor:")
    print(transformed)
    print("\nGradients:")
    print(grads)
    
    return x, transformed, grads

# Run the demonstration
demonstrate_differentiable_transform()
# %%
class ECDF(Layer):
    def __init__(self, alpha=0, beta=0, *args, **kwargs):
        super(ECDF, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta


    def call(self, inputs):
        x = inputs
        n = tf.size(x)

        unique_vals, unique_indices = tf.unique(x)
        counts = tf.math.bincount(unique_indices)
        cumulative_counts = tf.cumsum(counts)
        ecdf_vals = (
            (cumulative_counts - self.alpha) /
            (n + 1 - self.alpha - self.beta) 
        )

        indices = tf.searchsorted(unique_vals, x, side='right') - 1
        indices = tf.clip(indices, 0, tf.size(ecdf_vals) - 1)

        return ecdf_vals[indices]
    
    @staticmethod
    def unique(self, x):
        vals, indices = tf.argsort(x), tf.argsort(x)