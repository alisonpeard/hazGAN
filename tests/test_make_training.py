"""Test marginals.R script output

>> pytest tests/ -x
"""
import pytest
import warnings

try:
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    from environs import Env
    import matplotlib.pyplot as plt
    from statsmodels.stats.diagnostic import acorr_ljungbox

    env = Env()
    env.read_env(recurse=True)
    wd = env.str("TRAINDIR")
    testdir = os.path.dirname(os.path.abspath(__file__))

except Exception as e:
    pass


def test_imports():
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    from environs import Env
    import matplotlib.pyplot as plt
    from statsmodels.stats.diagnostic import acorr_ljungbox


def test_environment():
    assert env is not None
    assert wd is not None


@pytest.fixture
def metadata():
    """Training data we are testing GPD fit on"""
    return pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))


@pytest.fixture
def storms():
    """Training data we are testing GPD fit on"""
    return pd.read_parquet(os.path.join(wd, "storms.parquet"))


@pytest.fixture
def data():
    """Training data we are testing GPD fit on"""
    return xr.open_dataset(os.path.join(wd, "data.nc"))


@pytest.mark.parametrize('lag', [1])
def test_storm_extractor_autocorrelation(data, lag, alpha=0.1):
    """Test that the storm maxima are not autocorrelated
    
    (lag one only for now).
    """
    u10 = data.sel(channel="u10")['anomaly']
    storm_maxima = u10.max(dim=['lat', 'lon']).data
    res = acorr_ljungbox(storm_maxima, lags=lag, return_df=False)
    p = res.loc[lag, 'lb_pvalue']
    assert p >= alpha, (
        "Reject H0:independent storm maxima for lag {}. ".format(lag) +
        "p-value: {:.4f} ".format(p) + 
        "check storm_extractor in utils.R"
        )


def test_data_data_1940_2022_alignment(data, data_1940_2022):
    """Test that the data is a subset of data_1940_2022"""
    assert data.time.min() >= data_1940_2022.time.min()
    assert data.time.max() <= data_1940_2022.time.max()


    raise NotImplementedError("Test not implemented")