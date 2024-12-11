import numpy as np


class Empirical(object):
    """Empirical distribution object.
    
    Defaults to Weibull plotting positions, as recommended
    for hydrological data.
    
    Attributes:
    -----------
    x : array-like
        Data to fit empirical distribution to.
    alpha : float, optional (default=0)
        Determines plotting position type.
    beta : float, optional (default=0)
        Determines plotting position type.
    """
    def __init__(self, x, alpha=0, beta=0) -> None:
        self.x = np.sort(np.asarray(x))
        self.n = len(self.x)

        if self.n < 1:
            raise ValueError("'x' must have 1 or more non-missing values")
        
        self.alpha = alpha
        self.beta = beta
        self.cdf = self._cdf()
        self.quantile = self._quantile()


    def _cdf(self) -> callable:
        x = self.x
        n = self.n

        unique_vals, unique_indices = np.unique(x, return_inverse=True)
        counts = np.bincount(unique_indices)
        cumulative_counts = np.cumsum(counts)
        ecdf_vals = (
            (cumulative_counts - self.alpha) /
            (n + 1 - self.alpha - self.beta) 
        )

        def interpolator(query_points):
            indices = np.searchsorted(unique_vals, query_points, side='right') - 1
            indices = np.clip(indices, 0, len(ecdf_vals) - 1)
            return ecdf_vals[indices]

        return interpolator
    
    def _quantile(self) -> callable:
        """Empirical quantile function."""
        x = sorted(self.x)
        u = sorted(self.cdf(x))

        def interpolator(query_points):
            return np.interp(query_points, u, x)
        
        return interpolator