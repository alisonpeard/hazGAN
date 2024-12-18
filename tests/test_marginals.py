"""Test marginals.R script outputs are correct.

>> pytest tests/ -x

Notes:
------
To only use subset of storms data for a test:
@pytest.mark.parametrize('storms',
                         [('u10'), ('tp')],
                         indirect=['storms']) # keep for reference
"""
import pytest

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
    """Training data we are testing GPD fit on."""
    dataset = xr.open_dataset(os.path.join(wd, "data_1940_2022.nc"))
    dataset = dataset.rename({'msl': 'mslp'})
    dataset['mslp'] = -dataset['mslp']
    return dataset


@pytest.fixture
def metadata():
    """Training data we are testing GPD fit on."""
    return pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))


@pytest.fixture(params=['u10', 'mslp', 'tp'])
def storms(request):
    """Training data we are testing GPD fit on, add cols as needed."""
    field = request.param
    storms = pd.read_parquet(os.path.join(wd, "storms.parquet"))
    columns = ['time.{}', 'storm', 'grid', '{}', 'ecdf.{}', 'thresh.{}',
                'scale.{}', 'shape.{}', 'p.{}']
    columns = [col.format(field) for col in columns]
    storms = storms[columns]
    storms.columns = ['time', 'storm', 'grid', 'field', 'ecdf', 'thresh',
                'scale', 'shape', 'p']
    storms['storm'] = storms['storm'].astype(int)
    storms['grid'] = storms['grid'].astype(int) 
    storms['time'] = pd.to_datetime(storms['time'])
    storms['fieldname'] = [field] * len(storms)
    return storms


def test_storms_fixture(storms):
    """Check storms loaded correctly."""
    assert not storms.empty
    assert 'time' in storms.columns
    assert storms['fieldname'].nunique() == 1, "{}".format(storms['fieldname'].unique())


def test_storms_data_1940_2022_alignment(data_1940_2022, storms):
    """Test that wind values in storms.parquet haven't changed from
    input (data_1940_2022) during marginals.R script.
    """
    field = storms['fieldname'][0]
    data_out = storms.copy()
    data_out = data_out.set_index(['time', 'grid'])

    data_in = data_1940_2022.copy()
    data_in = data_in.rename({field: 'field'})
    medians = data_in['field'].groupby('time.month').median() 
    data_in['field'] = data_in['field'].groupby('time.month') - medians
    data_in = data_in[['grid', 'field']].to_dataframe().reset_index()
    data_in['time'] = data_in['time'].dt.date
    data_in = data_in.set_index(['time', 'grid'])
    intersection = data_in.join(data_out, how='right', lsuffix='_in', rsuffix='_out')

    assert np.isclose(intersection['field_in'], intersection['field_out']).all(), (
        f"Mismatch in {field} values between data_1940_2022.nc and storms.parquet"
    )


@pytest.mark.parametrize('lag, storms', [(1,'u10')], indirect=['storms'])
def test_storm_maxima_autocorrelation(storms, lag, alpha=0.1):
    """Test that the storm maxima are not autocorrelated.
    
    (lag one only for now).
    """
    data_out = storms.groupby('storm').apply(
        lambda x: pd.Series({
            'field': x['field'].max(),
            'time': x['time'][x['field'].idxmax()]
            }),
            include_groups=False
            )
    
    data_out['time'] = pd.to_datetime(data_out['time'])
    data_out = data_out.sort_values(by='time')
    data_out = data_out.set_index('time')

    storm_maxima = data_out['field'].to_numpy()
    res = acorr_ljungbox(storm_maxima, lags=lag, return_df=False)
    p = res.loc[lag, 'lb_pvalue']
    assert p >= alpha, (
        "Reject H0:independent storm maxima for lag {}. ".format(lag) +
        "p-value: {:.4f} ".format(p) + 
        "check storm_extractor in utils.R"
        )


def test_ecdf_values_strict(storms):
    """Test that the empirical CDF is within bounds (0, 1)"""
    from hazGAN.constants import TEST_YEAR

    storms = storms[storms['time'].dt.year != TEST_YEAR].copy()
    ecdf = storms['ecdf']
    assert ecdf.max() < 1, 'ecdf ≥ 1 found, {:.4f}'.format(ecdf.max())
    assert ecdf.min() > 0, 'ecdf ≤ 0 found {:.4f}'.format(ecdf.min())


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300, 302])
def test_ecdf_function_open_max(storms, cell):
    """Test that the empirical CDF is within bounds (0, 1)"""
    from hazGAN.statistics import ecdf

    test = storms[storms['grid'] == cell].copy()
    field = test['field']
    ecdf_function = ecdf(field)

    results = []
    for pow in [2, 4, 6, 8]:
        result = ecdf_function(10**pow)
        results.append(result)
    results = np.array(results)

    assert all(results < 1), (
        "ecdf ≥ 1 found, {}".format(max(results))
    )


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300, 302])
def test_ecdf_function_open_min(storms, cell):
    """Test that the empirical CDF is within bounds (0, 1)"""
    from hazGAN.statistics import ecdf

    test = storms[storms['grid'] == cell].copy()
    field = test['field']
    ecdf_function = ecdf(field)

    results = []
    for pow in [2, 4, 6, 8]:
        result = ecdf_function(10**-pow)
        results.append(result)
    results = np.array(results)

    assert all(results > 0), (
        "ecdf ≤ 0 found, {}".format(min(results))
    )
        


# takes a while... comment out for now
@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300, 302])
def test_ecdf(storms, cell, tol=1e-6):
    """Test that applying ecdf function recovers the 'ecdf' column"""
    from hazGAN.constants import TEST_YEAR
    from hazGAN.statistics import ecdf
    
    test = storms[storms['time'].dt.year != TEST_YEAR].copy()
    test = test[test['grid'] == cell].copy()

    field = test['field']
    test['ecdf_test'] = ecdf(field)(field)
    test['difference'] = test['ecdf'] - test['ecdf_test']
    assert np.isclose(test['difference'], 0, atol=tol).all(), (
        "{}".format(abs(test['difference']).max())
    )


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300])
def test_quantile(storms, cell, tol=1e-6):
    """Test that applying quantile function recovers the field column"""
    from hazGAN.constants import TEST_YEAR
    from hazGAN.statistics import quantile
    
    test = storms[storms['time'].dt.year != TEST_YEAR].copy()
    test = test[test['grid'] == cell].copy()

    field = test['field']
    ecdf = test['ecdf']
    test['quantile'] = quantile(field)(ecdf)
    test['difference'] = test['field'] - test['quantile']
    assert np.isclose(test['difference'], 0, atol=tol).all(), (
        "{}".format(abs(test['difference']).max())
    )


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300, 302])
def test_gpd_params(storms, cell, tol=1e-6) -> None:
    """Check only one parameter set per grid cell."""
    test  = storms[storms['grid'] == cell].copy()
    scale = test['scale']
    shape = test['shape']
    loc   = test['thresh']

    assert np.isclose(scale.min(), scale.max(), atol=tol)
    assert np.isclose(shape.min(), shape.max(), atol=tol)
    assert np.isclose(loc.min(), loc.max(), atol=tol)


@pytest.mark.parametrize('cell', [1, 5, 20, 40, 100, 200, 300, 302])
def test_semigpd(storms, cell, tol=1) -> None:
    """Test semi-parametric fit reasonable close."""
    from hazGAN.statistics import GenPareto
    
    test  = storms[storms['grid'] == cell].copy()
    fieldname = test['fieldname'].iloc[0]
    field = test['field']
    scale = test['scale'].iloc[0]
    shape = test['shape'].iloc[0]
    loc   = test['thresh'].iloc[0]

    gpd_fit = GenPareto(field, loc, scale, shape)
    field_u = gpd_fit.forward(field)
    field_x = gpd_fit.inverse(field_u)
    difference = field - field_x

    assert abs(difference).sum() < tol, (
        "Total difference: {:.4f}".format(abs(difference).sum()) +
        " for {}".format(fieldname) +
        " in grid cell {}".format(cell)
    )


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

    # load test fixtures for dev
    from hazGAN.constants import TEST_YEAR

    FIELD = "u10"
    data = xr.open_dataset(os.path.join(wd, "data.nc"))
    data = data.sel(field=[FIELD])

    data_1940_2022 = xr.open_dataset(os.path.join(wd, "data_1940_2022.nc"))
    data_1940_2022 = data_1940_2022.rename({'msl': 'mslp'})
    data_1940_2022['mslp'] = -data_1940_2022['mslp']
    metadata = pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))

    storms = pd.read_parquet(os.path.join(wd, "storms.parquet"))
    columns = ['time.{}', 'storm', 'grid', '{}', 'scdf.{}', 'thresh.{}',
                'scale.{}', 'shape.{}', 'p.{}']
    columns = [col.format(FIELD) for col in columns]
    storms = storms[columns]
    storms.columns = ['time', 'storm', 'grid', 'field', 'scdf', 'thresh',
                'scale', 'shape', 'p']
    storms['storm'] = storms['storm'].astype(int)
    storms['grid'] = storms['grid'].astype(int) 
    storms['time'] = pd.to_datetime(storms['time'])
    storms['fieldname'] = [FIELD] * len(storms)

    # %%
    def test_invPIT(storms, data):
        from hazGAN.statistics import GenPareto
        from hazGAN.statistics import invPITDataset
        from hazGAN import make_grid
        
        # apply invPIT to the data
        x_array = invPITDataset(data)
        field = data.field.values[0]
        n, h, w, _ = x_array.shape  

        x_array = make_grid(x_array)

        for cell in range(h * w):
            gridcell = storms[storms['grid'] == cell].reset_index(drop=True)
            loc = gridcell['thresh'][0]
            scale = gridcell['scale'][0]
            shape = gridcell['shape'][0]
            x = gridcell['field'][:]
            u = gridcell['scdf'][:]

            genpareto = GenPareto(x, loc, scale, shape)
            x_cell = genpareto.inverse(u)

            x_data = x_array.where(x_array['grid']==0, drop=True)['x'].data
            x_data = x_data.squeeze()

            assert np.allclose(x_data, x_cell), "Mismatch in cell {} for {}.".format(cell, field)
    # %%