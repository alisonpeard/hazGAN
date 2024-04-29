"""Doing all this in R now"""

import numpy as np
from .base import *


def probability_integral_transform(dataset):
    """Transform data to uniform distribution using ecdf."""
    n, h, w, c = dataset.shape
    assert c == 1, "single channel only"
    dataset = dataset[..., 0].reshape(n, h * w)

    uniform, _ = semiparametric_cdf(
        dataset, fit_tail=False
    )  # fully parametric by default (maybe change that)
    parameters = gev_cdf(dataset)

    uniform = uniform.reshape(n, h, w, 1)
    parameters = parameters.reshape(h, w, 3)
    return uniform, parameters


def gev_cdf(dataset):
    assert dataset.ndim == 2, "Requires 2 dimensions"
    x = dataset.copy()
    n, J = np.shape(x)
    shape = np.empty(J)
    loc = np.empty(J)
    scale = np.empty(J)

    for j in (pbar := tqdm(range(J))):
        pbar.set_description("Fitting GEV to marginals.")
        shape[j], loc[j], scale[j] = genextreme.fit(x[:, j], method="MLE")
    parameters = np.stack([shape, loc, scale], axis=-1)
    return parameters


def inv_probability_integral_transform(marginals, params):
    """Transform uniform marginals to original distributions, using quantile function of GEV."""
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape
    marginals = marginals.reshape(n, h * w, c)
    assert params.shape == (
        h,
        w,
        3,
        c,
    ), "Marginals and parameters have different dimensions."
    params = params.reshape(h * w, 3, c)

    quantiles = []
    for channel in range(c):
        q = np.array(
            [
                genextreme.ppf(marginals[:, j, channel], *params[j, ..., channel])
                for j in range(h * w)
            ]
        ).T
        quantiles.append(q)

    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles
