
import pytest

try:
    import numpy as np
    import xarray as xr
    from collections import Counter
    from hazGAN.io import label_data
except Exception as e:
    pass


def test_imports():
    import numpy as np
    import xarray as xr
    from hazGAN.io import label_data
    from collections import Counter


def test_label_data():
    x = np.arange(0, 20)
    data = xr.DataArray(x, dims='x')
    dataset = xr.Dataset({'maxwind': data})
    label_ratios = {'5': .25, '10': .25, '15': .25, '999': .25}

    labels = label_data(dataset, label_ratios)
    counts = Counter(list(labels.to_numpy()))


    expected_counts = {1: 5, 2: 5, 3: 5, 4: 5}
    assert counts == expected_counts, "Expected {}, got {}.".format(expected_counts, counts)