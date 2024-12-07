# %%
import numpy as np
from scipy.stats import genpareto
import warnings

if __name__ == "__main__":
    from base import *
else:
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
    marginals = inv_gumbel(marginals).numpy() if gumbel_margins else marginals

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


# %% Unit tests
# 1. f(f^{-1}(u)) = u
# 2. with and without Gumbel transform g(x) = -log(-log(x))

# load data.nc
if __name__ == "__main__":
    import os
    import xarray as xr
    from environs import Env
    import matplotlib.pyplot as plt

    CHANNEL = 0

    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    x = data['uniform'].data[..., [CHANNEL]]
    y = data['anomaly'].data[..., [CHANNEL]]
    params = data['params'].data[..., [CHANNEL]] # lat, lon, param, channel

    print("x.shape: {}\ny.shape: {}\nparams.shape: {}".format(x.shape, y.shape, params.shape))
    # %%
    test = inv_probability_integral_transform(x, x, y, params=params, gumbel_margins=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(x[0, ..., 0])
    axs[0].set_title("Uniform")
    axs[1].imshow(test[0, ..., 0])
    axs[1].set_title("Inverse PIT")

    fig.colorbar(axs[0].imshow(x[0, ..., 0]), ax=axs[0])
    fig.colorbar(axs[1].imshow(test[0, ..., 0]), ax=axs[1])

    # %%
    assert np.allclose(x, test), "Inverse PIT failed."
    # %%
