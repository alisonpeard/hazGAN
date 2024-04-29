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
        Uniform marginals.
    x : np.array
        Data that original quantiles were calculated from.
    y : np.array
        Corresponding empirical distribution estimates.
    params : np.array
        Parameters of fitted GPD distribution.
    """
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape

    marginals = inv_gumbel(marginals) if gumbel_margins else marginals

    assert x.shape[1:] == (h,w,c,), f"Marginals and x have different dimensions: {x.shape[1:]} != {h, w, c}."
    assert y.shape[1:] == (h,w, c,), f"Marginals and y have different dimensions: {y.shape[1:]} != {h, w, c}."
    assert (x.shape[0] == tf.shape(y)[0]), f"x and y have different dimensions: {x.shape[0]} != {y.shape[0]}."

    marginals = marginals.reshape(n, h * w, c)
    x = x.reshape(len(x), h * w, c)
    y = y.reshape(len(y), h * w, c)

    if params is not None:
        assert params.shape == (h,w,3,c,), "Marginals and parameters have different dimensions."
        params = params.reshape(h * w, 3, c)

    quantiles = []
    for channel in range(c):
        if params is None:
            q = np.array(
            [
                empirical_quantile(
                    marginals[:, j, channel],
                    x[:, j, channel], y[:, j, channel]
                    )
                    for j in range(h * w)
                ]
            ).T
        else:
            thresh = params[..., 0, channel]
            q = np.array(
            [
                empirical_quantile(
                    marginals[:, j, channel],
                    x[:, j, channel],
                    y[:, j, channel],
                    params[j, ..., channel]
                )
                for j in range(h * w)
            ]
        ).T
        quantiles.append(q)
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
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


