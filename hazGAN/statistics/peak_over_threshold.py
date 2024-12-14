# %%
import numpy as np

if __name__ == "__main__":
    from base import *
    from empirical import quantile, semiparametric_quantile
else:
    from .base import *
    from .empirical import quantile, semiparametric_quantile


def inv_probability_integral_transform(
        u:np.ndarray,
        x:np.ndarray,
        theta:np.ndarray=None,
        gumbel_margins:bool=False
    ) -> np.ndarray:
    """
    Transform uniform marginals to original distributions via inverse interpolation of empirical CDF.
    
    Parameters
    ----------
    u : np.ndarray
        Uniform marginals with shape [n, h, w, c] or [n, h * w, c]
    x : np.ndarray
        Original data for quantile calculation
    theta : np.ndarray, optional
        Parameters of fitted Generalized Pareto Distribution (GPD)
    gumbel_margins : bool, default False
        Whether to apply inverse Gumbel transform
    
    Returns
    -------
    np.ndarray
        Transformed marginals with original shape
    """
    u = inv_gumbel(u).numpy() if gumbel_margins else u

    assert x.shape[1:] == u.shape[1:], (
        f"Marginal dimensions mismatch: {u.shape[1:]} != {x.shape[1:]}"
    )

    # flatten along spatial dimensions
    original_shape = u.shape
    if u.ndim == 4:
        n, h, w, c = u.shape
        hw = h * w
        u = u.reshape(n, hw, c)
        x = x.reshape(len(x), hw, c)
        if theta is not None:
            theta = theta.reshape(hw, 3, c)
            theta = theta.transpose(1, 0, 2)
    elif u.ndim == 3:
        n, hw, c = u.shape
    else:
        raise ValueError(
            "Uniform marginals must have dimensions [n, h, w, c] or [n, h * w, c]."
            )    

    # vectorised numpy implementation
    def transform(x, u, theta, i, c):
        x_i = x[:, i, c]
        u_i = u[:, i, c]
        theta_i = theta[:, i, c] if theta is not None else None
        return (
            semiparametric_quantile(x_i, theta_i)(u_i)
            if theta is not None
            else quantile(x_i)(u_i)
        )

    quantiles = np.array([
        transform(x, u, theta, i, channel)
        for i in range(hw) for channel in range(c) 
    ])
    quantiles = quantiles.T
    quantiles = quantiles.reshape(*original_shape)
    return quantiles


# %% Unit tests

# load data.nc
if __name__ == "__main__":
    import os
    import xarray as xr
    from environs import Env
    import matplotlib.pyplot as plt
    from hazGAN.utils import TEST_YEAR

    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    mask = data['time.year'] != TEST_YEAR
    test_mask = data['time.year'] == TEST_YEAR
    test = data.sel(time=test_mask)
    data = data.sel(time=mask)

    # %%
    u = data['uniform'].data
    x = data['anomaly'].data
    theta = data['params'].data
    u_test = test['uniform'].data

    print("x.shape: {},\ntheta.shape: {}".format(x.shape, theta.shape))
    plt.imshow(u[0, ..., 0])
    
    # %% test fit 
    THETA = theta
    CHANNEL = 2
    fitted = inv_probability_integral_transform(u, x, theta=THETA, gumbel_margins=False)

    difference = fitted - x
    difference_per_sample = np.abs(difference).sum(axis=(1, 2, 3))
    idxmax = np.argmax(difference_per_sample)
    print("Sample with max difference: {}".format(idxmax))

    difference_per_cell = np.abs(difference).sum(axis=(0, 3)).ravel()
    idxmax_cell = np.unravel_index(np.argmax(difference_per_cell), difference_per_cell.shape)
    print("Cell with max difference: {}".format(idxmax_cell))

    sample = 779 # idxmax
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    im = axs[0].imshow(x[sample, ..., CHANNEL])
    axs[0].set_title("Anomaly")
    plt.colorbar(im, ax=axs[0], fraction=0.046, shrink=0.8)

    im = axs[1].imshow(u[sample, ..., CHANNEL])
    axs[1].set_title("Uniform")
    plt.colorbar(im, ax=axs[1], fraction=0.046, shrink=0.8)

    im = axs[2].imshow(fitted[sample, ..., CHANNEL])
    axs[2].set_title("Inverse PIT")
    plt.colorbar(im, ax=axs[2], fraction=0.046, shrink=0.8)
    

    im = axs[3].imshow(difference[sample, ..., CHANNEL], cmap="Spectral_r")
    plt.colorbar(im, ax=axs[3], fraction=0.046, shrink=0.8)
    axs[3].set_title("Difference")
    
    difference_flat = difference.flatten()
    print("Max difference: {:.4f}".format(np.abs(difference_flat).max()))
    print("Mean difference: {:.4f}".format(np.abs(difference_flat).mean()))

    # %% test predictions
    test = inv_probability_integral_transform(u_test, x, theta=THETA, gumbel_margins=False)

    # %%
    from hazGAN import Empirical, ecdf, quantile

    grid = 300
    channel = 0
    n, h, w, c = x.shape
    u_test = u_test.reshape(len(u_test), h*w, c)[:, grid, channel]
    x = x.reshape(n, h*w, c)[:, grid, channel]

    # %%
    quantile(x)(u_test).max()
    u_test.max()
    # %%
    from hazGAN import semiparametric_quantile, semiparametric_cdf

    theta = theta.reshape(h*w, 3, 3)[grid, :, channel]
    semiparametric_quantile(x, theta)(u_test).max()
    semiparametric_cdf(x, theta)(x).max()
# %%
