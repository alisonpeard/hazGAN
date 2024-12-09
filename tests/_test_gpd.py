# pytest tests/ -x
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
def data():
    """Training data we are testing GPD fit on"""
    return xr.open_dataset(os.path.join(wd, "data.nc"))


@pytest.fixture
def metadata():
    """Training data we are testing GPD fit on"""
    return pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))


def test_storm_maxima(data, metadata):
    """Test that the storm maxima match those used in R script."""
    before = metadata.groupby('storm').apply(lambda x: pd.Series({
        'u10': x['u10'].max(),
        'time': x['time'][x['u10'].idxmax()]
    }))
    before = before.set_index('time', drop=True)

    u10 = data.sel(channel="u10")['anomaly']
    after = u10.max(dim=['lat', 'lon'])
    after = after.to_dataframe('u10').drop(columns=['channel'])
    after = after.sort_values('time')

    # assert before.equals(after)
    comparison = pd.concat([before, after], axis=1)
    comparison.columns = ['before', 'after']
    comparison['difference'] = comparison['after'] - comparison['before']
    assert comparison['difference'].sum() == 0

    num_misaligned_maxima = comparison['difference'].isnull().sum()
    if num_misaligned_maxima > 0:
        warnings.warn("Found {} misaligned storm maxima".format(num_misaligned_maxima))


@pytest.mark.parametrize('lag', [1])
def test_storm_maxima_autocorrelation(data, lag, alpha=0.1):
    """Test that the storm maxima are not autocorrelated
    
    (lag one only for now).
    """
    u10 = data.sel(channel="u10")['anomaly']
    storm_maxima = u10.max(dim=['lat', 'lon']).data
    res = acorr_ljungbox(storm_maxima, lags=lag, return_df=False)
    p = res.loc[lag, 'lb_pvalue']
    assert p >= alpha, (
        "Reject H0:independent storm maxima for lag {}.".format(lag) +
        "p-value: {:.4f}".format(p)
        )

# what should I test?
# storm domain maxima Ljung-box test
# anomaly vanilla ecdf == uniform
# f(uniform) ~= anomaly, where f is the inverse of the (semiparametric) ecdf
# test the fit of exceedences to GPD