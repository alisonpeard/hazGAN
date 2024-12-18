# pytest tests/ -x
import pytest

try:
    import os
    import numpy as np
    import xarray as xr
    import pandas as pd
    from environs import Env
    import matplotlib.pyplot as plt
    from hazGAN.utils import rescale, frobenius, get_similarities

    env = Env()
    env.read_env(recurse=True)
    wd = env.str("TRAINDIR")
    testdir = os.path.dirname(os.path.abspath(__file__))

except Exception as e:
    pass


def test_imports():
    import os
    import numpy as np
    import xarray as xr
    import pandas as pd
    from environs import Env
    import matplotlib.pyplot as plt
    from hazGAN.utils import rescale, frobenius, get_similarities


def test_environment():
    assert env is not None
    assert wd is not None, "TRAINDIR not found in .env file"


@pytest.fixture
def template():
    return np.load(os.path.join(testdir, "data/windbomb.npy"))


@pytest.fixture
def data_1940_2022():
    """Training data we are testing GPD fit on"""
    return xr.open_dataset(os.path.join(wd, "data_1940_2022.nc"))

@pytest.fixture
def data():
    """Training data we are testing GPD fit on"""
    return xr.open_dataset(os.path.join(wd, "data.nc"))


@pytest.fixture
def metadata():
    """The storm metadata from R script"""
    return pd.read_parquet(os.path.join(wd, "storms_metadata.parquet"))


def test_metadata_preserves_u10(data_1940_2022, metadata):
    a = metadata[['time', 'u10']].copy()
    a['time'] = pd.to_datetime(a['time'])
    a = a.set_index('time', drop=True)

    b = data_1940_2022['u10']
    monthly = b.groupby('time.month').median() 
    b = b.groupby('time.month') - monthly
    b = b.max(dim=['lat', 'lon']).to_dataframe('u10').drop(columns=['month'])
    b.index = pd.to_datetime(b.index)

    c = a.join(b, how='left', lsuffix='_a', rsuffix='_b')
    c['difference'] = c['u10_a'] - c['u10_b']
    c['difference'].sum()

    assert c['difference'].sum() == 0, \
        'storms_metadata.parquet should match max winds in original data.'


def test_storms_parquet_nonans():
    """Test that there are no NaNs in storms.parquet"""
    storms = pd.read_parquet(os.path.join(wd, "storms.parquet"))
    params = ['scale', 'shape', 'thresh', 'loc']
    nans_forbidden = [col for col in storms.columns if not any(map(col.__contains__, params))]
    for column in nans_forbidden:
        null_count = storms[column].isnull().sum()
        if null_count > 0:
            print(f"{column}: {null_count} null entries")
    assert storms[nans_forbidden].isnull().sum().sum() == 0, "NaNs found in storms.parquet."


def test_data_nc_nonans(data):
    """Test that there are no NaNs in the data.nc"""
    data = data.drop_vars('params')
    nulls_total = 0
    for var in data.data_vars:
        null_count = data[var].isnull().sum().data.item()
        if null_count > 0:
            print(f"{var}: {null_count} null entries")
            nulls_total += null_count
    assert nulls_total == 0, "NaNs found in data.nc"


def test_windbomb_exists():
    windbomb = np.load(os.path.join(wd, "windbomb.npy"))
    assert windbomb is not None


def test_windbomb_values(template):
    """Test that the windbomb values are the same as the template."""
    windbomb = np.load(os.path.join(wd, "windbomb.npy"))

    assert windbomb is not None
    assert np.allclose(windbomb, template)

    windbomb = rescale(windbomb)
    template = rescale(template)
    assert frobenius(windbomb, template) > 0.9


@pytest.mark.parametrize("threshold", [0.8, 0.85, 0.9])
def test_windbombs_removed(template, threshold, tmp_path):
    """Test no more wind bombs in training data"""
    data = xr.open_dataset(os.path.join(wd, "data.nc"))
    data['u10'] = (data['anomaly'] + data['medians']).sel(field='u10')
    similarities = get_similarities(data, template)

    exceeded = similarities >= threshold
    num_exceeded = np.sum(exceeded)
    
    if num_exceeded > 0:
        problematic_indices = np.where(exceeded)[0]
        problematic_times = data.time.data[problematic_indices]

        max_idx = np.argmax(similarities)
        data['u10'].isel(time=max_idx).plot()
        plot_path = tmp_path / f'threshold_{threshold}_max_similarity.png'
        plt.savefig(plot_path)
        plt.close()
        
        pytest.fail(f"{num_exceeded} samples exceeded threshold {threshold} at time: {problematic_times}" +
                    f"\nPlot saved at {plot_path}" + 
                    f"'\nOpen from terminal with `open {plot_path}`")
    
    assert num_exceeded == 0, f"{num_exceeded} samples exceeded threshold {threshold}"
    