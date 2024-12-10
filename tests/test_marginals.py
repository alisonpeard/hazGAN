"""Test marginals.R script outputs are correct.

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
def data_1940_2022():
    """Training data we are testing GPD fit on"""
    return xr.open_dataset(os.path.join(wd, "data_1940_2022.nc"))


@pytest.fixture
def metadata():
    """Training data we are testing GPD fit on"""
    return pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))


@pytest.fixture
def storms():
    """Training data we are testing GPD fit on, add cols as needed"""
    storms = pd.read_parquet(os.path.join(wd, "storms.parquet"))
    storms = storms[['time.u10', 'storm', 'grid', 'u10', 'ecdf.u10']]
    storms.columns = ['time', 'storm', 'grid', 'u10', 'ecdf']
    storms['storm'] = storms['storm'].astype(int)
    storms['grid'] = storms['grid'].astype(int) 
    storms['time'] = pd.to_datetime(storms['time'])
    return storms



def test_storms_data_1940_2022_alignment(storms, data_1940_2022):
    """Test that wind values in storms.parquet haven't changed from
    data_1940_2022 during marginals.R script.
    """
    data_in = data_1940_2022.copy()
    medians = data_in['u10'].groupby('time.month').median() 
    data_in['u10'] = data_in['u10'].groupby('time.month') - medians
    data_in = data_in[['grid', 'u10']].to_dataframe().reset_index()
    data_in['time'] = data_in['time'].dt.date
    data_in = data_in.set_index(['time', 'grid'])

    data_out = storms.copy()
    data_out = data_out.set_index(['time', 'grid'])

    intersection = data_in.join(data_out, how='right', lsuffix='_in', rsuffix='_out')
    assert np.isclose(intersection['u10_in'], intersection['u10_out']).all()


@pytest.mark.parametrize('lag', [1])
def test_storm_extractor_autocorrelation(storms, lag, alpha=0.1):
    """Test that the storm maxima are not autocorrelated
    
    (lag one only for now).
    """
    data_out = storms.groupby('storm').apply(
        lambda x: pd.Series({
            'u10': x['u10'].max(),
            'time': x['time'][x['u10'].idxmax()]
            }),
            include_groups=False
            )
    
    data_out['time'] = pd.to_datetime(data_out['time'])
    data_out = data_out.sort_values(by='time')
    data_out = data_out.set_index('time')

    storm_maxima = data_out['u10'].to_numpy()
    res = acorr_ljungbox(storm_maxima, lags=lag, return_df=False)
    p = res.loc[lag, 'lb_pvalue']
    assert p >= alpha, (
        "Reject H0:independent storm maxima for lag {}. ".format(lag) +
        "p-value: {:.4f} ".format(p) + 
        "check storm_extractor in utils.R"
        )


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300])
def test_ecdf_gets_same_result(storms, cell, tol=0.01):
    """Test that applying ecdf function recovers the 'ecdf' column"""
    from hazGAN.utils import TEST_YEAR

    def ecdf(x:pd.Series):
        n = len(x)
        x = x.rank(method="average").to_numpy()
        return x / (n + 1)
    
    test = storms[storms['time'].dt.year != TEST_YEAR].copy()
    test = test[test['grid'] == cell].copy()
    test['ecdf_test'] = ecdf(test['u10'])
    test['difference'] = test['ecdf'] - test['ecdf_test']
    assert np.isclose(test['difference'], 0, atol=tol).all()



# %% [dev section]
if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    import datetime as dt

    from environs import Env
    import matplotlib.pyplot as plt
    
    env = Env()
    env.read_env(recurse=True)
    wd = env.str("TRAINDIR")

    data = xr.open_dataset(os.path.join(wd, "data.nc"))
    data_1940_2022 = xr.open_dataset(os.path.join(wd, "data_1940_2022.nc"))
    metadata = pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))
    storms = pd.read_parquet(os.path.join(wd, "storms.parquet"))
    storms = storms[['time.u10', 'storm', 'grid', 'u10', 'ecdf.u10']]
    storms.columns = ['time', 'storm', 'grid', 'u10', 'ecdf']
    storms['storm'] = storms['storm'].astype(int)
    storms['grid'] = storms['grid'].astype(int) 




    # FULL DATAFRAME NOT WORKING
    # storms_wide = storms.copy().pivot(index='time', columns='grid', values='u10')
    # storms_wide = storms_wide.sort_index()
    # storms_wide = storms_wide.apply(ecdf, axis=0)

    # storms_long = storms_wide.melt(ignore_index=False, value_name='ecdf')
    # storms_long = storms_long.set_index('grid', append=True)

    # storms = storms.set_index(['time', 'grid']) 

    # test = storms.join(storms_long, how='left', lsuffix='_in')
    # test['difference'] = test['ecdf_in'] - test['ecdf']

    # TAKE A TEST CELL FIRST

# %% 