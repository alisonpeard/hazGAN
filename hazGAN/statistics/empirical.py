import numpy as np
from scipy.stats import genpareto

# the following functions are simple wrappers for the classes below
def ecdf(x:np.ndarray) -> callable:
    """Simple wrapper to mimic R ecdf."""
    return Empirical(x).forward


def quantile(x:np.ndarray) -> callable:
    """Simple wrapper to mimic R ecdf but for quantile function."""
    return Empirical(x).inverse


def semiparametric_cdf(x, params) -> callable:
    """Semi-parametric CDF."""
    loc, scale, shape = params
    return GenPareto(x, loc, scale, shape).forward


def semiparametric_quantile(u, params) -> callable:
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
        tail_mask = x >= self.loc
        tail_x = x[tail_mask]

        tail_fit = genpareto.cdf(tail_x, self.shape, loc=self.loc, scale=self.scale)
        tail_u = 1 - (1 - loc_u) * (1 - tail_fit)
        u[tail_mask] = tail_u

        return u


    def _semiquantile(self, uniform) -> np.ndarray:
        # empirical base
        x = self.quantile(uniform)

        # parametric tail
        loc_u = self.ecdf(self.loc)
        tail_mask = uniform >= loc_u
        tail_u = uniform[tail_mask]

        tail_u = 1 - ((1 - tail_u) / (1 - loc_u))
        tail_x = genpareto.ppf(tail_u, self.shape, loc=self.loc, scale=self.scale)

        x[tail_mask] = tail_x

        return x
    

