# %%
import numpy as np
import xarray as xr
from functools import partial

if __name__ == "__main__":
    import base
    from empirical import quantile, ecdf
    from empirical import semiparametric_quantile as semiparametric_quantile0
    from empirical import semiparametric_cdf as semiparametric_cdf0
else:
    from . import base
    from .empirical import quantile, ecdf
    from .empirical import semiparametric_quantile as semiparametric_quantile0
    from .empirical import semiparametric_cdf as semiparametric_cdf0


def invPIT(
        u:np.ndarray,
        x:np.ndarray,
        theta:np.ndarray=None,
        margins:str="uniform",
        distribution:str="genpareto"
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
    margins : str, optional (default = False)
        Whether to apply inverse transform for reduced variate margins
    
    Returns
    -------
    np.ndarray
        Transformed marginals with same shape as input u
    """
    cdf = getattr(base, "inv_" + margins)
    u = cdf(u)

    # check x for nans
    if np.any(np.isnan(x)):
        raise ValueError("'x' contains NaNs. Cannot compute quantiles.")

    semiparametric_quantile = partial(
        semiparametric_quantile0, distribution=distribution
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

    # vectorised numpy transform
    distns = ["genpareto", "genpareto", "genpareto"]
    def transform(x, u, theta, i, c):
        x_i = x[:, i, c]
        u_i = u[:, i, c]
        theta_i = theta[:, i, c] if theta is not None else None
        distn = distns[c]
        return (
            semiparametric_quantile(x_i, theta_i, distn)(u_i)
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


def empiricalPIT(
        x:np.ndarray,
        xref:np.ndarray=None,
    ) -> np.ndarray:
    """
    Transform original distribution to uniform marginals via empirical CDF.
    
    Parameters
    ----------
    x : np.ndarray
        Original data for quantile calculation
    xref : np.ndarray, optional (default = None)
        Reference data for empirical CDF calculation
    
    Returns
    -------
    np.ndarray
        Transformed marginals with same shape as input u
    """
    if xref is None:
        xref = x

    original_shape = x.shape
    if x.ndim == 4:
        n, h, w, c = x.shape
        hw = h * w
        x = x.reshape(n, hw, c)
        xref = xref.reshape(len(xref), hw, c)
    elif x.ndim == 3:
        n, hw, c = x.shape
    else:
        raise ValueError(
            "Data must have dimensions [n, h, w, c] or [n, h * w, c]."
            )    

    # vectorised numpy transform
    def transform(x, xref, i, c):
        x_i = x[:, i, c]
        x_r = xref[:, i, c]
        return (
            ecdf(x_r)(x_i)
        )

    probs = np.array([
        transform(x, xref, i, channel)
        for i in range(hw) for channel in range(c) 
    ])

    probs = probs.T
    probs = probs.reshape(*original_shape)

    return probs

