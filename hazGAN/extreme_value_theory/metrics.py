import numpy as np
import tensorflow as tf
# from tensorflow_probability import distributions
from .base import *


@tf.function
def chi2metric(samples, nbins=20):
    """
    H0: samples are uniformly distributed
    
    Want to minimise prob
    """
    shape = tf.shape(samples)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    samples = tf.reshape(samples, (b * h * w, c))

    expected = tf.cast(b * h * w, dtype=tf.float32)
    expected = tf.cast(expected / nbins, dtype=tf.float32)

    max = tf.reduce_max(samples)
    min = tf.reduce_min(samples)
    bins = tf.histogram_fixed_width_bins(samples, [min, max], nbins=nbins)
    
    one_hot_bins = tf.one_hot(bins, depth=nbins)
    counts = tf.reduce_sum(one_hot_bins, axis=0)
    teststat = tf.reduce_sum((counts - expected) ** 2 / expected)
    # chi2dist = distributions.Chi2(df=nbins-1)
    # prob = chi2dist.cdf(teststat)
    return teststat # of chisq(nbins) values being smaller than observed


@tf.function
def chi_loss(real, fake):
    """
    Calculate average pairwise (spatial) chi score between real and fake data across each channel.
    
    Args:
    -----
    real : tf.Tensor, [b, h, w, c]
        Real data.
    fake : tf.Tensor, [b, h, w, c]
        Generated data.
    """
    
    def compute_chi_diff(i):
        return channel_chi_diff(real, fake, i)

    c = tf.shape(real)[-1]
    chi_diff = tf.vectorized_map(compute_chi_diff, tf.range(c))
    return tf.reduce_mean(chi_diff)


@tf.function
def channel_chi_diff(real, fake, channel=0):
    # TODO: this isn't very numerically stable
    real = real[..., channel]
    fake = fake[..., channel]
    ecs_real = pairwise_extremal_coeffs(real)
    ecs_fake = pairwise_extremal_coeffs(fake)
    diff = ecs_real - ecs_fake
    return tf.sqrt(tf.reduce_mean(tf.square(diff)))


@tf.function
def pairwise_extremal_coeffs(uniform):
    """Calculate extremal coefficients for each pair of pixels across single channel."""
    assert (
        len(uniform.shape) == 3
            ), "Function all_extremal_coeffs fors for 3-dimensional tensors only."
    shape = tf.shape(uniform)
    n, h, w = shape[0], shape[1], shape[2]
    uniform = tf.reshape(uniform, (n, h * w))
    frechet = inv_frechet(uniform)

    minima = minner_product(tf.transpose(frechet), frechet)

    n = tf.cast(n, frechet.dtype)
    minima = tf.cast(minima,frechet.dtype)
    ecs = tf.math.divide_no_nan(n, minima) # returns zero if denominator is zero
    return ecs


@tf.function
def minner_product(a, b):
    "Use broadcasting to get sum of pairwise minima."
    x = tf.reduce_sum(
        tf.minimum(
            tf.expand_dims(a, axis=-1),
            tf.expand_dims(b, axis=0)),
        axis=1
    )
    return x


def test_minner_product():
    x = np.array([[1, 2], [1, 1]])
    assert np.array_equal(minner_product(x.T, x), np.array([[2, 2], [2, 3]])), f"{minner_product(x.T, x)}"


# Other metrics
def kstest_uniform(x):
    """KS test statistic for uniformity. Small => uniform."""
    n = tf.shape(x)[0]
    sorted_x = tf.sort(x)
    indices = tf.range(n, dtype=tf.float32) / tf.cast(n, dtype=tf.float32)
    ecdf = tf.cast(indices, dtype=tf.float32)
    d = tf.reduce_max(tf.abs(ecdf - sorted_x))
    return d


# Pairwise correlations
def get_all_correlations(x):
    """Get pairwise correlations for each pixel in a single channel."""
    n, h, w = x.shape
    k = h * w
    x = x.reshape(n, k)
    corrs = np.ones([k, k])
    for i in range(k):
        for j in range(i - 1):
            corr = np.corrcoef(x[:, i], x[:, j])[0, 1]
            corrs[i, j] = corr
            corrs[j, i] = corr
    return corrs

# Cross-channel correlations
def get_extremal_corrs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across D-dimensional uniform data."""
    _, _, _, d = marginals.shape
    coefs = get_extremal_coeffs_nd(marginals, sample_inds)
    return {key: d - val for key, val in coefs.items()}


def get_extremal_coeffs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across D-dimensional uniform data."""
    n, h, w, d = marginals.shape
    data = marginals.reshape(n, h * w, d)
    data = data[:, sample_inds, :]
    frechet = inv_frechet(data)
    ecs = {}
    for i in range(len(sample_inds)):
        ecs[sample_inds[i]] = raw_extremal_coeff_nd(frechet[:, i, :])
    return ecs


def raw_extremal_coeff_nd(frechets):
    n, d = frechets.shape
    minima = np.min(frechets, axis=1)  # minimum for each row
    minima = np.sum(minima)
    if minima > 0:
        theta = n / minima
    else:
        print("Warning: all zeros in minima array.")
        theta = d
    return theta





########################################################################################
# TODO: OLD
# Potential loss functions
# @tf.function
def kstest_loss(x):
    """KS test statistic for uniformity. Small => uniform."""
    shape = tf.shape(x)
    n, h, w, c = shape[0], shape[1], shape[2], shape[3]
    sorted_x = tf.sort(x, axis=0)
    indices = tf.range(n, dtype=tf.float32) / tf.cast(n, dtype=tf.float32)
    ecdf = tf.reshape(tf.cast(indices, dtype=tf.float32), (n, 1, 1, 1))
    d = tf.reduce_max(tf.abs(ecdf - sorted_x), axis=0)
    return d

# def get_extremal_correlations(marginals, sample_inds):
#     coeffs = get_extremal_coeffs(marginals, sample_inds)
#     coors = {indices: 2 - coef for indices, coef in coeffs.items()}
#     return coors


# def get_extremal_coeffs(marginals, sample_inds):
#     assert (
#         len(marginals.shape) == 3
#     ), f"Requires marginals of rank 3, provided with rank {len(data.shape)} marginals."
#     data = tf.cast(marginals, dtype=tf.float32)
#     n, h, w = tf.shape(data)
#     data = tf.reshape(data, [n, h * w])
#     data = tf.gather(data, sample_inds, axis=1)
#     frechet = inv_frechet(data)
#     ecs = {}
#     for i in range(len(sample_inds)):
#         for j in range(i):
#             ecs[sample_inds[i], sample_inds[j]] = raw_extremal_coefficient(
#                 frechet[:, i], frechet[:, j]
#             ).numpy()
#     return ecs


# def raw_extremal_coefficient(frechet_x, frechet_y):
#     """Where x and y have been transformed to their Fréchet marginal distributions.

#     ..[1] Max-stable process and spatial extremes, Smith (1990) §4
#     """
#     n = tf.shape(frechet_x)[0]
#     assert n == tf.shape(frechet_y)[0]
#     n = tf.cast(n, dtype=tf.float32)
#     minima = tf.reduce_sum(tf.math.minimum(frechet_x, frechet_y))
#     if tf.greater(tf.reduce_sum(minima), 0):
#         theta = n / minima
#     else:
#         tf.print("Warning: all zeros in minima array.")
#         theta = 2
#     return theta



