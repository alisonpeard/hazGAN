"""Helper functions for running evtGAN in TensorFlow."""
import os
import yaml
import numpy as np
import xarray as xr
from shapely.geometry import Point

def res2str(res):
    return f"{res[0]}x{res[1]}"


def notify(title, subtitle, message):
    """Display OSX system notification with title and subtitle."""
    os.system("""
              osascript -e 'display notification "{}" with title "{}" subtitle "{}" beep'
              """.format(message, title, subtitle))


def load_config_from_yaml(configfile:str) -> dict:
    """Load configuration from a YAML file."""
    with open(configfile, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config = {key: value['value'] for key, value in config.items()}
    return config


def rescale(x:np.ndarray) -> np.ndarray:
    return (x - x.min() / (x.max() - x.min()))


def rescale_vector(x:np.ndarray) -> np.ndarray:
    return (x - x.min(axis=(1, 2), keepdims=True)) / (x.max(axis=(1, 2), keepdims=True) - x.min(axis=(1, 2), keepdims=True))


def frobenius(test:np.ndarray, template:np.ndarray) -> float:
    sum_ = np.sum(template * test)
    norms = np.linalg.norm(template) * np.linalg.norm(test)
    similarities = sum_ / norms
    return similarities


def frobenius_vector(test:np.ndarray, template:np.ndarray) -> np.ndarray:
    sum_ = np.sum(template * test, axis=(1, 2))
    norms = np.linalg.norm(template) * np.linalg.norm(test, axis=(1, 2))
    similarities = sum_ / norms
    return similarities


def get_similarities(ds:xr.Dataset, template:np.ndarray) -> np.ndarray:
    """Get similarities between a template and dataset."""
    template = rescale(template)
    tensor = ds['u10'].data
    tensor = rescale_vector(tensor)
    similarities = frobenius_vector(tensor, template)
    
    return similarities # np.array(similarities)


def op2idx(ops:dict, data:np.ndarray, extent:list):
    h, w = data.shape

    lons = np.linspace(extent[0], extent[1], w)
    lats = np.linspace(extent[2], extent[3], h)

    coords = np.array([Point(lon, lat) for lon in lons for lat in lats])

    op_idx = {}
    for op, loc in ops.items():
        idx = np.argmin([coord.distance(Point(loc)) for coord in coords])
        op_idx[op] = idx

    return op_idx
    
def diff(x, d=1):
    """Difference a (time series) array."""
    return x[d:] - x[:-d]


def translate_indices(i, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = np.argwhere(indices.reshape(dims[0], dims[1]) == i)
    return tuple(*map(tuple, x))


def translate_indices_r(i, j, dims=(18, 22)):
    indices = np.arange(0, dims[0] * dims[1], 1)
    x = indices.reshape(dims[0], dims[1])[i, j]
    return x


def sliding_windows(x, size, step=1):
    n, *_ = x.shape
    window_indices = sliding_window_indices(size, n, step)
    return x[window_indices, ...]


def sliding_window_indices(size, n, step=1):
    windows = []
    i = 0
    for i in range(0, n - size, step):
        windows.append(np.arange(i, i+size, 1))
    return np.array(windows)

