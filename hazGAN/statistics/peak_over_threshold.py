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
    import numpy as np
    from empirical import GenPareto, Empirical

    def mape(x, y):
        if np.isclose(x, 0).any():
            x = x + 1e-6
        return np.mean(np.abs((x - y) / x)) * 100

    # %% big comparison of different fits
    differences = []
    for ds in [data, data_test]:
        difference_dict = {}
        for FIELD in ['u10', 'tp', 'mslp']:
            maxdiffs_emp = []
            maxdiffs_semi = []
            maxdiffs_gpd = []
            for CELL in range(1, 18*22):
                test = data.sel(field=FIELD)
                test = test.where(test['grid'] == CELL, drop=True)
                test

                # test GPD -> forward -> inverse
                scale = test.sel(param='scale')['params'].data.item()
                shape = test.sel(param='shape')['params'].data.item()
                loc   = test.sel(param='loc')['params'].data.item()
                x     = test['anomaly'].data.squeeze()

                gpd_fit = GenPareto(x, loc, scale, shape)
                fit  = Empirical(x)

                # pure empirical
                field_u = fit.forward(x)
                field_x = fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_emp.append(maxdiff)

                # pure GPD
                field_u = gpd_fit.forward(x)
                field_x = gpd_fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_gpd.append(maxdiff)

                # empirical + GPD
                field_u = fit.forward(x)
                field_x = gpd_fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_semi.append(maxdiff)
            
            difference_dict[FIELD] = {
                'empirical': np.sum(maxdiffs_emp),
                'gpd': np.sum(maxdiffs_gpd),
                'semi': np.sum(maxdiffs_semi)
                }
        differences.append(difference_dict)
    differences = {'train': differences[0], 'test': differences[1]}

    # %%
    reform = {(outerKey, innerKey): values for outerKey, innerDict in differences.items() for innerKey, values in innerDict.items()}
    differences = pd.DataFrame(reform).T
