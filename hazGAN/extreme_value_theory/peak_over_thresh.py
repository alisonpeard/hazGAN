"""Doing all this in R now"""

import numpy as np
from scipy.stats import genpareto
import warnings
from .base import *


def inv_probability_integral_transform(marginals, x, y, params=None, gumbel_margins=False):
    """
    Transform uniform marginals to original distributions, by inverse-interpolating ecdf.
    
    Args:
    -----
    marginals : np.array
        Uniform marginals with dimensions [n, h, w, c] or [n, h * w, c].
    x : np.array
        Data that original quantiles were calculated from.
    y : np.array
        Corresponding empirical distribution estimates.
    params : np.array
        Parameters of fitted GPD distribution.
    """
    marginals = inv_gumbel(marginals) if gumbel_margins else marginals

    assert x.shape[1:] == marginals.shape[1:], f"Marginals and x have different dimensions: {marginals.shape[1:]} != {x.shape[1:]}."
    assert y.shape[1:] == marginals.shape[1:], f"Marginals and y have different dimensions: {marginals.shape[1:]} != {y.shape[1:]}."
    assert (x.shape[0] == tf.shape(y)[0]), f"x and y have different number of samples: {x.shape[0]} != {y.shape[0]}."

    original_shape = marginals.shape
    if marginals.ndim == 4:
        n, h, w, c = marginals.shape
        hw = h * w
        marginals = marginals.reshape(n, hw, c)
        x = x.reshape(len(x), hw, c)
        y = y.reshape(len(y), hw, c)
        if params is not None:
            params = params.reshape(hw, 3, c)
    elif marginals.ndim == 3:
        n, hw, c = marginals.shape
    else:
        raise ValueError("Marginals must have dimensions [n, h, w, c] or [n, h * w, c].")      

    quantiles = []
    for channel in range(c):
        if params is None:
            q = np.array(
            [
                empirical_quantile(
                    marginals[:, j, channel],
                    x[:, j, channel], y[:, j, channel]
                    )
                    for j in range(hw)
                ]
            ).T
        else:
            q = np.array(
            [
                empirical_quantile(
                    marginals[:, j, channel],
                    x[:, j, channel],
                    y[:, j, channel],
                    params[j, ..., channel]
                )
                for j in range(hw)
            ]
        ).T
        quantiles.append(q)
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(*original_shape)
    return quantiles


# NOTE: Currently using R for the below but might be useful again later
#---------------------------------------------------------------------
# def decluster_array(x: np.array, thresh: float, r: int):
#     """Decluster array of exceedences."""
#     if (r is None) or (r == 0):
#         warnings.warn(f"decluster_array only returns indices of exceedences for r={r}.")
#         return np.where(x > thresh)[0]
#     exceedences = x > thresh
#     clusters = identify_clusters(exceedences, r)
#     cluster_maxima = []
#     for cluster in clusters:
#         argmax = cluster[np.argmax(x[cluster])]
#         cluster_maxima.append(argmax)
#     return cluster_maxima


# def identify_clusters(x: np.array, r: int):
#     """Identify clusters of Trues separated by at least r Falses."""
#     clusters = []
#     cluster_no = 0
#     clusters.append([])
#     false_counts = 0
#     for i, val in enumerate(x):
#         if false_counts == r:
#             clusters.append([])
#             cluster_no += 1
#             false_counts = 0
#         if val:
#             clusters[cluster_no].append(i)
#         else:
#             false_counts += 1
#     clusters = [cluster for cluster in clusters if len(cluster) > 0]
#     return clusters


# def probability_integral_transform(
#     dataset, prior=None, thresholds=None, fit_tail=False, decluster=None
# ):
#     """Transform data to uniform distribution using ecdf."""
#     warnings.warn("This hasn't been updated in ages because using R.")
#     n, h, w, c = dataset.shape
#     assert c == 1, "single channel only"
#     dataset = dataset[..., 0].reshape(n, h * w)

#     if fit_tail is True:
#         assert thresholds is not None, "Thresholds must be supplied if fitting tail."
#         thresholds = thresholds.reshape(h * w)

#     uniform, parameters = semiparametric_cdf(
#         dataset, prior, thresholds, fit_tail=fit_tail, decluster=decluster
#     )
#     uniform = uniform.reshape(n, h, w, 1)
#     parameters = parameters.reshape(h, w, 3)
#     return uniform, parameters


