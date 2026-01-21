"""
Functions for transforming data to other distributions using empirical cdf.
"""
import numpy as np
from warnings import warn


def frechet(uniform):
    """Fréchet"""
    return -1 / np.log(uniform)


def inverted_frechet(uniform):
    """Inverted Fréchet RV is also exponentially distributed."""
    return -np.log(uniform)


def exp(uniform):
    """Exponential"""
    return -np.log(1 - uniform)


def inv_exp(uniform):
    """Inverse exponential"""
    return 1 - np.exp(-uniform)


def gumbel(uniform):
    """uniform -> Gumbel(0, 1)"""
    maxval = np.max(uniform)
    if maxval >= 1:
        raise ValueError("Some uniform >= 1")
    return -np.log(-np.log(uniform))


def inv_gumbel(x):
    """Gumbel(0, 1) -> uniform"""
    return np.exp(-np.exp(-x))


def uniform(u):
    """Identity function for uniform marginals."""
    return u

def inv_uniform(x):
    """Identity function for uniform marginals."""
    return x

def gaussian(u):
    """uniform -> standard Gaussian"""
    from scipy.special import erfinv
    return np.sqrt(2) * erfinv(2 * u - 1)

def inv_gaussian(x):
    """standard Gaussian -> uniform"""
    from scipy.stats import norm
    return norm.cdf(x)


def rescaled(u):
    """Identity function for any marginals."""
    return u

def inv_rescaled(x):
    """Identity function for any marginals."""
    return x