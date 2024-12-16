"""
Functions for transforming data to other distributions using empirical cdf.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import genextreme, genpareto
from tqdm import tqdm
import warnings
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable



def frechet(uniform):
    """Fréchet"""
    return -1 / tf.math.log(uniform)


def inverted_frechet(uniform):
    """Inverted Fréchet RV is exponentially distributed."""
    return -tf.math.log(uniform)


def exp(uniform):
    """Exponential"""
    return -tf.math.log(1 - uniform)


def inv_exp(uniform):
    """Inverse exponential"""
    return 1 - tf.math.exp(-uniform)


def gumbel(uniform):
    """Make Gumbel"""
    maxval = tf.reduce_max(uniform) # .numpy()
    if maxval == 1:
        warnings.warn("Values == 1 found, scaling by 1e-6")
        uniform *= 1 - 1e-6
    if maxval > 1:
        raise ValueError("Some uniform > 1")
    return -tf.math.log(-tf.math.log(uniform))


def inv_gumbel(x):
    """Gumbel -> uniform"""
    return tf.math.exp(-tf.math.exp(-x))


# def _semigpd_tail(x, params):
#     """Tail CDF for semi-parametric GPD."""
#     loc, scale, shape = params
#     gpd_cdf = genpareto.cdf(x, shape, loc=loc, scale=scale)
#     cdf = 
#     raise NotImplementedError("GPD not implemented.")


# def _invgpd(uniform, pthresh, params):
#     """
#     Inverse of (1.3) H&T for $\ksi\leq 0$ and upper tail.

#     Args:
#     -----
#     uniform : array-like
#         Uniform random variable.
#     pthresh : float
#         Threshold percentile.
#     params : tuple
#         (shape, loc, scale)
#     """
#     shape, loc, scale = params
#     x = loc + (scale / shape) * (((1 - uniform) / (1 - pthresh)) ** (-shape) - 1)
#     return x


# def empirical_quantile(uniform, x, y, params=None):
#     """Used in inverse PIT function.

#     (Semi)empirical quantile/percent/point function.

#     x [was] a vector of interpolated uniform quantiles of data (usually 100,000)
#     Now x and y are data that original quantiles were calculated from, where x
#     is data and y corresponding densities. #!ARE THEY?
#     """
#     #!!!CHECK THIS!!!
#     x = sorted(x)

#     if (uniform.max() - uniform.min()) == 0.0:
#         return np.array([-999] * len(uniform))  # no data proxy

#     if uniform.max() >= 1:
#         warnings.warn("Some uniform >= 1.")
#         uniform *= 1 - 1e-6

#     quantiles = np.interp(uniform, sorted(y), sorted(x))
#     if params is not None:
#         loc = params[1]
#         f_thresh = np.interp(loc, sorted(x), sorted(y))
#         uniform_tail = uniform[uniform >= f_thresh]
#         quantiles_tail = _upper_ppf(uniform_tail, f_thresh, params)
#         quantiles[uniform >= f_thresh] = quantiles_tail
#     return quantiles
