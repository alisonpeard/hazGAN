"""
Functions for transforming data to other distributions using empirical cdf.
"""

import numpy as np
import tensorflow as tf
from scipy.stats import genextreme, genpareto
from tqdm import tqdm
import warnings
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def rank(x):
    if x.std() == 0:
        ranked = np.array([len(x) / 2 + 0.5] * len(x))
    else:
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        ranked = ranks + 1
    return ranked


def ecdf(x):
    return rank(x) / (len(x) + 1)


def frechet(uniform):
    return -1 / tf.math.log(uniform) # possible place for nans


def inv_frechet(uniform):
    """Inverted Fréchet RV is exponentially distributed."""
    return -tf.math.log(uniform)


def exp(uniform):
    return -tf.math.log(1 - uniform)


def gumbel(uniform):
    maxval = tf.reduce_max(uniform).numpy()
    if maxval == 1:
        warnings.warn("Some uniform == 1, scaling by 1e-6")
        uniform *= 1 - 1e-6
    if maxval > 1:
        raise ValueError("Some uniform > 1")
    return -tf.math.log(-tf.math.log(uniform))


def inv_gumbel(x):
    return tf.math.exp(-tf.math.exp(-x))


# TODO: fix uniform
def upper_ppf(uniform, f_thresh, params):
    """
    Inverse of (1.3) H&T for $\ksi\leq 0$ and upper tail.

    Args:
    -----
    params : tuple
        (shape, loc, scale)
    """
    shape, loc, scale = params
    x = loc + (scale / shape) * (((1 - uniform) / (1 - f_thresh)) ** (-shape) - 1)
    return x


def empirical_quantile(uniform, x, y, params=None):
    """Used in inverse PIT function.

    (Semi)empirical quantile/percent/point function.

    x [was] a vector of interpolated uniform quantiles of data (usually 100,000)
    Now x and y are data that original quantiles were calculated from, where x
    is data and y corresponding densities.
    """
    x = sorted(x)

    if (uniform.max() - uniform.min()) == 0.0:
        return np.array([-999] * len(uniform))  # no data proxy

    if uniform.max() >= 1:
        warnings.warn("Some uniform >= 1.")
        uniform *= 1 - 1e-6

    quantiles = np.interp(uniform, sorted(y), sorted(x))
    if params is not None:
        loc = params[1]
        f_thresh = np.interp(loc, sorted(x), sorted(y))
        uniform_tail = uniform[uniform >= f_thresh]
        quantiles_tail = upper_ppf(uniform_tail, f_thresh, params)
        quantiles[uniform >= f_thresh] = quantiles_tail
    return quantiles
