import numpy as np
from scipy.stats import genpareto

# the following functions are simple wrappers for the classes below
def ecdf(x:np.ndarray, *args, **kwargs) -> callable:
    """Simple wrapper to mimic R ecdf."""
    return Empirical(x).forward


def quantile(x:np.ndarray, *args, **kwargs) -> callable:
    """Simple wrapper to mimic R ecdf but for quantile function."""
    return Empirical(x).inverse


def semiparametric_cdf(x, params, *args, **kwargs) -> callable:
    """Semi-parametric CDF."""
    loc, scale, shape = params
    return GenPareto(x, loc, scale, shape).forward


def semiparametric_quantile(u, params, *args, **kwargs) -> callable:
    """Semi-parametric quantile."""
    loc, scale, shape = params
    return GenPareto(u, loc, scale, shape).inverse


# class definitions
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

    Methods:
    --------
    forward : callable
        Empirical CDF.
    inverse : callable
        Empirical quantile function
    """
    def __init__(self, x, alpha=0, beta=0) -> None:
        self.x = np.sort(np.asarray(x))
        self.n = len(self.x)

        if self.n < 1:
            raise ValueError("'x' must have 1 or more non-missing values")
        
        self.alpha = alpha
        self.beta = beta

        self.ecdf = self._ecdf()
        self.quantile = self._quantile()

        self.forward = self._ecdf()
        self.inverse = self._quantile()


    def _ecdf(self) -> callable:
        x = self.x
        n = self.n

        unique_vals, unique_indices = np.unique(x, return_inverse=True)
        counts = np.bincount(unique_indices)
        cumulative_counts = np.cumsum(counts)
        ecdf_vals = (
            (cumulative_counts - self.alpha) /
            (n + 1 - self.alpha - self.beta) 
        )

        assert max(ecdf_vals) < 1, "ECDF values exceed 1."

        def interpolator(query_points):
            indices = np.searchsorted(unique_vals, query_points, side='right') - 1
            indices = np.clip(indices, 0, len(ecdf_vals) - 1)
            return ecdf_vals[indices]

        return interpolator
    
    def _quantile(self) -> callable:
        """Empirical quantile function."""
        x = sorted(self.x)
        u = sorted(self.ecdf(x))

        def interpolator(query_points):
            return np.interp(query_points, u, x)
        
        return interpolator


class GenPareto(Empirical):
    """Semi-empirical GPD distribution object."""
    def __init__(self, x,
                 loc=0, scale=1, shape=1,
                 alpha=0, beta=0) -> None:
        super().__init__(x, alpha, beta)

        self.loc = loc
        self.scale = scale
        self.shape = shape

        self.ecdf = super()._ecdf()
        self.quantile = super()._quantile()

        self.forward = self._semicdf
        self.inverse = self._semiquantile


    def _semicdf(self, x) -> np.ndarray:
        """(1.3) H&T for $\ksi\leq 0$ and upper tail."""
        # empirical base
        u = self.ecdf(x)
        loc_u = self.ecdf(self.loc)

        # parametric tail
        tail_mask = x > self.loc
        tail_x = x[tail_mask]

        tail_fit = genpareto.cdf(tail_x, self.shape, loc=self.loc, scale=self.scale)
        tail_u = 1 - (1 - loc_u) * (1 - tail_fit)
        u[tail_mask] = tail_u

        try:
            assert np.isfinite(u).all(), "Non-finite values in CDF."
            assert not np.isnan(u).any(), "NaN values in CDF."

            if self.shape < 0:
                assert (u <= 1).all(), "CDF values above 1."
                assert (u > 0).all(), "CDF values ≤ 0."
            else:
                assert (u >= 0).all(), "CDF values below 0."
                assert (u < 1).all(), "CDF values ≥ 1."
        
        except AssertionError as e:
            print(e)
            print("x: ", min(x), max(x))
            print("u: ", min(u), max(u))
            print("loc: ", self.loc)
            print("scale: ", self.scale)
            print("shape: ", self.shape)
            raise e

        return u


    def _semiquantile(self, u) -> np.ndarray:
        # empirical base
        x = self.quantile(u)

        # parametric tail
        loc_u = self.ecdf(self.loc)
        tail_mask = u > loc_u
        tail_u = u[tail_mask]

        tail_u = 1 - ((1 - tail_u) / (1 - loc_u))
        tail_x = genpareto.ppf(tail_u, self.shape, loc=self.loc, scale=self.scale)

        x[tail_mask] = tail_x

        try:
            assert np.isfinite(x).all(), "Non-finite values in quantile function."
            assert not np.isnan(x).any(), "NaN values in quantile function."

        except AssertionError as e:
            print(e)
            print("u: ", min(u), max(u))
            print("x: ", min(x), max(x))
            print("loc: ", self.loc)
            print("scale: ", self.scale)
            print("shape: ", self.shape)
            print("tail_fit min: ", min(tail_x))
            print("tail_fit max: ", max(tail_x))
            print("multiplicative constant: ", 1 - ((1 - tail_u) / (1 - loc_u)))
            raise e

        return x
    

