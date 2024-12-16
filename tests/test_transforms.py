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
    columns = ['time.{}', 'storm', 'grid', '{}', 'scdf.{}', 'thresh.{}',
                'scale.{}', 'shape.{}', 'p.{}']
    columns = [col.format(field) for col in columns]
    storms = storms[columns]
    storms.columns = ['time', 'storm', 'grid', 'field', 'scdf', 'thresh',
                'scale', 'shape', 'p']
    storms['storm'] = storms['storm'].astype(int)
    storms['grid'] = storms['grid'].astype(int) 
    storms['time'] = pd.to_datetime(storms['time'])
    storms['fieldname'] = [field] * len(storms)
    return storms


@pytest.fixture(params=['u10', 'mslp', 'tp'])
def data(request):
    """Training data we are testing GPD fit on."""
    field = request.param
    dataset = xr.open_dataset(os.path.join(wd, "data.nc"))
    return dataset.sel(field=[field])

@pytest.mark.parametrize('storms, data',
                         [('u10', 'u10'),
                          ('tp', 'tp'),
                          ('mslp', 'mslp')],
                         indirect=['storms', 'data'])
def test_invPIT(storms, data):
    """Test vectorised inverse PIT matches GenPareto."""
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

        x_data = x_array.where(x_array['grid'] == cell, drop=True)['x'].data
        x_data = x_data.squeeze()

        diff = np.abs(x_data - x_cell)

        assert np.allclose(x_data, x_cell), (
            "Mismatch in cell {} for {}. ".format(cell, field) + 
            "Max difference: {}.".format(diff.max())
        )
        
