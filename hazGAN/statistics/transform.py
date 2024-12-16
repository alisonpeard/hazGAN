# %%
import numpy as np
import xarray as xr

if __name__ == "__main__":
    from base import *
    from empirical import quantile, semiparametric_quantile
else:
    from .base import *
    from .empirical import quantile, semiparametric_quantile



def invPITDataset(ds:xr.Dataset, theta_var:str="params",
                  u_var:str="uniform", x_var:str="anomaly",
                  gumbel_margins:bool=False) -> xr.DataArray:
    """
    Wrapper of invPIT for xarray.Dataset.
    """
    u = ds[u_var].values
    x = ds[x_var].values
    theta = ds[theta_var].values if theta_var in ds else None

    x_inv = invPIT(u, x, theta, gumbel_margins)
    x_inv = xr.DataArray(x_inv, dims=ds[x_var].dims, coords=ds[x_var].coords)

    return x_inv



def invPIT(
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
    theta : np.ndarray, optional (default = None)
        Parameters of fitted Generalized Pareto Distribution (GPD)
    gumbel_margins : bool, optional (default = False)
        Whether to apply inverse Gumbel transform
    
    Returns
    -------
    np.ndarray
        Transformed marginals with same shape as input u
    """
    u = inv_gumbel(u).numpy() if gumbel_margins else u

    # assert x.shape[1:] == u.shape[1:], (
    #     f"Marginal dimensions mismatch: {u.shape[1:]} != {x.shape[1:]}"
    # )

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

    # vectorised numpy transform
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


# %% Developing tests
if __name__ == "__main__":
    import os
    import pandas as pd
    import xarray as xr
    from environs import Env
    import matplotlib.pyplot as plt
    from hazGAN.utils import TEST_YEAR
    from hazGAN.xarray import make_grid

    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    
    storms = pd.read_parquet(os.path.join(datadir, "storms.parquet"))
    storms['time.u10'] = pd.to_datetime(storms['time.u10'])
    storms['time.tp'] = pd.to_datetime(storms['time.tp'])
    storms['time.mslp'] = pd.to_datetime(storms['time.mslp'])
    storms = storms[storms['time.u10'].dt.year != TEST_YEAR]
    storms_test = storms[storms['time.u10'].dt.year == TEST_YEAR]

    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    mask = data['time.year'] != TEST_YEAR
    test_mask = data['time.year'] == TEST_YEAR

    data_test = data.sel(time=test_mask)
    data = data.sel(time=mask)

    # %% test alignment with cellwise
