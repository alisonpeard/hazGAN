# wrap R functions to use in Python scripts
import rpy2.robjects as robjects

PATH = "../scripts/data_processing/utils.R"


class R:
    """
    Lazy loading of R functions for use in Python scripts (optional but useful).
    
    Example usage:
    --------------
    >>> ecdf = R.get('ecdf')
    >>> c = R.c()
    """
    _r_sourced = False
    _global_r_env = None

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

