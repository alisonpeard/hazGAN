#%% wrap R functions to use in Python scripts
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


PATH = "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/scripts/data_processing/utils.R"


class R:
    """
    Lazy loading of R functions for use in Python scripts (optional but useful).
    
    Example usage:
    --------------
    >>> from hazGAN import R
    >>> ecdf = R.get('ecdf')
    >>> c = R.c()
    """
    _r_sourced    = False
    _global_r_env = None
    _extRemes     = None

    @classmethod
    def source(cls, path=PATH):
        """
        Lazily source R code only when first accessed.
        Ensures R code is loaded only once.
        """
        if not cls._r_sourced:
            robjects.r.source(path)
            cls._global_r_env = robjects.globalenv
            cls._r_sourced = True
        return cls._global_r_env

    @classmethod
    def get(cls, function_name, path=PATH):
        """
        Retrieve a specific R function after sourcing the R code.
        
        :param function_name: Name of the R function to retrieve
        :return: The requested R function
        """
        env = cls.source(path)
        return env[function_name]
    
    @staticmethod
    def c():
        """
        Convenience method to retrieve the R 'c' function.
        """
        return robjects.FloatVector
    
    @staticmethod
    def _import(package:str):
        """
        Import an R package.
        """
        return importr(package)
    

    @classmethod
    def taildependence(cls, x, y, u, **kwargs):
        """
        Calculate tail dependence using extRemes package.

        If X and Y are stochastically independent then chi(u)=1-u,
        and chibar=0. If X=Y, then chi(u)=chi=1. If U=V, then
        chibar=1. If chi=0, then chibar<1.
        
        Parameters:
        -----------
        x, y : numpy.ndarray
            Input arrays for bivariate tail dependence
        u : float
            Threshold value
        **kwargs : dict
            Additional arguments to pass to taildep function
            
        Returns:
        --------
        dict
            Dictionary containing tail dependence coefficients
        """
        # Import package if not already imported
        if cls._extRemes is None:
            cls._extRemes = cls._import("extRemes")
        
        # Convert numpy arrays to R vectors
        x_r = robjects.FloatVector(x.flatten())
        y_r = robjects.FloatVector(y.flatten())
        u_r = robjects.FloatVector([u])
        
        # Call R function
        result = cls._extRemes.taildep(x_r, y_r, u_r, **kwargs)
        names  = ["chi", "chibar"]
        
        # Convert R result to Python
        return dict(zip(names, list(result)))


# %% tail dependence dev
if __name__ == "__main__":
    import numpy as np
    # Generate sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = 0.7 * x + np.random.normal(0, 0.5, 1000)
    
    # Calculate tail dependence
    result = R.taildependence(x, y, u=0.9)
    print(result)

# %%