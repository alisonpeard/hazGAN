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
    maxval = np.max(uniform) # .numpy()
    if maxval == 1:
        warn("Values == 1 found, scaling by 1e-6")
        uniform *= 1 - 1e-6
    if maxval > 1:
        raise ValueError("Some uniform > 1")
    return -np.log(-np.log(uniform))


def inv_gumbel(x):
    """Gumbel(0, 1) -> uniform"""
    return np.exp(-np.exp(-x))
