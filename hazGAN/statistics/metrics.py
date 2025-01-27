# %%
import numpy as np
from .base import *

def chi_rmse(real, fake):
    c = real.shape[1]
    chi_diff = np.zeros(c)
    for channel in range(c):
        real_channel = real[:, channel, :, :]
        fake_channel = fake[:, channel, :, :]
        chi_diff[channel] = chi_diff_1d(real_channel, fake_channel)
    return np.mean(chi_diff)

def chi_diff_1d(real, fake):
    ecs_real = pairwise_extremal_coeffs(real)
    ecs_fake = pairwise_extremal_coeffs(fake)
    diff = ecs_real - ecs_fake
    return np.sqrt(np.mean(np.square(diff)))


def pairwise_extremal_coeffs(uniform):
    """Calculate extremal coefficients for each pair of pixels across single channel."""
    assert (
        len(uniform.shape) == 3
            ), "Function all_extremal_coeffs fors for 3-dimensional tensors only."
    n, h, w = uniform.shape
    uniform = np.reshape(uniform, (n, h * w))
    frechet = inverted_frechet(uniform)
    minima = minner_product(frechet.T, frechet)
    n = float(n)
    minima = minima.astype(float)
    ecs = np.zeros_like(minima)
    ecs = np.divide(n, minima, out=ecs, where=minima != 0, dtype=float)
    return ecs


def minner_product(a, b):
    "Use broadcasting to get sum of pairwise minima."
    x = np.sum(
            np.minimum(
                np.expand_dims(a, axis=-1),
                np.expand_dims(b, axis=0)),
            axis=1
        )
    return x


def maxer_product(a, b):
    "Use broadcasting to get sum of pairwise maxima."
    x = np.sum(
            np.maximum(
                np.expand_dims(a, axis=-1),
                np.expand_dims(b, axis=0)),
            axis=1
        )
    return x


def test_minner_product():
    x = np.array([[1, 2], [1, 1]])
    assert np.array_equal(minner_product(x.T, x), np.array([[2, 2], [2, 3]])), f"{minner_product(x.T, x)}"

# %%
# @tf.function
# def chi2metric(samples, nbins=20):
#     """
#     H0: samples are uniformly distributed
    
#     Want to minimise prob
#     """
#     shape = tf.shape(samples)
#     b, h, w, c = shape[0], shape[1], shape[2], shape[3]
#     samples = tf.reshape(samples, (b * h * w, c))

#     expected = tf.cast(b * h * w, dtype=tf.float32)
#     expected = tf.cast(expected / nbins, dtype=tf.float32)

#     max = tf.reduce_max(samples)
#     min = tf.reduce_min(samples)
#     bins = tf.histogram_fixed_width_bins(samples, [min, max], nbins=nbins)
    
#     one_hot_bins = tf.one_hot(bins, depth=nbins)
#     counts = tf.reduce_sum(one_hot_bins, axis=0)
#     teststat = tf.reduce_sum((counts - expected) ** 2 / expected)
#     # chi2dist = distributions.Chi2(df=nbins-1)
#     # prob = chi2dist.cdf(teststat)
#     return teststat # of chisq(nbins) values being smaller than observed


# # Other metrics
# def kstest_uniform(x):
#     """KS test statistic for uniformity. Small => uniform."""
#     n = tf.shape(x)[0]
#     sorted_x = tf.sort(x)
#     indices = tf.range(n, dtype=tf.float32) / tf.cast(n, dtype=tf.float32)
#     ecdf = tf.cast(indices, dtype=tf.float32)
#     d = tf.reduce_max(tf.abs(ecdf - sorted_x))
#     return d


# # Pairwise correlations
# def get_all_correlations(x):
#     """Get pairwise correlations for each pixel in a single channel."""
#     n, h, w = x.shape
#     k = h * w
#     x = x.reshape(n, k)
#     corrs = np.ones([k, k])
#     for i in range(k):
#         for j in range(i - 1):
#             corr = np.corrcoef(x[:, i], x[:, j])[0, 1]
#             corrs[i, j] = corr
#             corrs[j, i] = corr
#     return corrs

# # Cross-channel correlations
# def get_extremal_corrs_nd(marginals, sample_inds):
#     """Calculate extremal coefficients across D-dimensional uniform data."""
#     _, _, _, d = marginals.shape
#     coefs = get_extremal_coeffs_nd(marginals, sample_inds)
#     return {key: d - val for key, val in coefs.items()}


def get_extremal_coeffs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across D-dimensional uniform data."""
    n, h, w, d = marginals.shape
    data = marginals.reshape(n, h * w, d)
    data = data[:, sample_inds, :]
    frechet = inverted_frechet(data)
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
    return float(theta)

