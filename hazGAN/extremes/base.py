"""
Functions for transforming data to other distributions using empirical cdf.
"""
import numpy as np
import tensorflow as tf
from scipy.stats import genextreme, genpareto
from tqdm import tqdm
import warnings
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class Empirical(object):
    """Empirical distribution object. Defaults to Weibull
    
    Attributes:
    -----------
    x : array-like
        Data to fit empirical distribution to.
    alpha : float, optional (default=0)
        Determine plotting position type.
    beta : float, optional (default=0)
        Determine plotting position type.
    """
    def __init__(self, x, alpha=0, beta=0) -> None:
        self.x = np.sort(np.asarray(x))
        self.n = len(self.x)

        if self.n < 1:
            raise ValueError("'x' must have 1 or more non-missing values")
        
        self.alpha = alpha
        self.beta = beta
        self.cdf = self._cdf()
        self.quantile = self._quantile()


    def _cdf(self) -> callable:
        x = self.x
        n = self.n

        unique_vals, unique_indices = np.unique(x, return_inverse=True)
        counts = np.bincount(unique_indices)
        cumulative_counts = np.cumsum(counts)
        ecdf_vals = (
            (cumulative_counts - self.alpha) /
            (n + 1 - self.alpha - self.beta) 
        )

        def interpolator(query_points):
            indices = np.searchsorted(unique_vals, query_points, side='right') - 1
            indices = np.clip(indices, 0, len(ecdf_vals) - 1)
            return ecdf_vals[indices]

        return interpolator
    
    def _quantile(self) -> callable:
        """Empirical quantile function."""
        x = sorted(self.x)
        u = sorted(self.cdf(x))

        def interpolator(query_points):
            return np.interp(query_points, u, x)
        
        return interpolator


def ecdf(x: np.ndarray) -> callable:
    """Simple wrapper to mimic R ecdf."""
    return Empirical(x).cdf


def quantile(x: np.ndarray) -> callable:
    """Simple wrapper to mimic R ecdf."""
    return Empirical(x).quantile


def frechet(uniform):
    return -1 / tf.math.log(uniform) # possible place for nans


def inv_frechet(uniform):
    """Inverted Fréchet RV is exponentially distributed."""
    return -tf.math.log(uniform)


def exp(uniform):
    return -tf.math.log(1 - uniform)


def gumbel(uniform):
    maxval = tf.reduce_max(uniform) # .numpy()
    if maxval == 1:
        warnings.warn("Values == 1 found, scaling by 1e-6")
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
    is data and y corresponding densities. #!ARE THEY?
    """
    #!!!CHECK THIS!!!
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
