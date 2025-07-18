"""
Improved input-output methods to flexibly load from pretraining and training sets.
"""
# %%
import os
import gc
import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
import dask.array as da
from collections import Counter

from .constants import TEST_YEAR


__all__ = ['print_if_verbose', 'label_data', 'sample_dict', 'check_validity', 'make_xr_grid', 'load_xr_data']


def print_if_verbose(string:str, verbose=True) -> None:
    """Self-explanatory"""
    if verbose:
        print(string)


def one_hot(a:np.ndarray) -> np.ndarray:
    categories, inverse = np.unique(a, return_inverse=True)
    encoded = np.zeros((a.size, categories.size))
    encoded[np.arange(a.size), inverse] = 1
    return encoded


def encode_strings(ds:xr.Dataset, variable:str) -> ArrayLike:
    """One-hot encode a string variable"""
    # check that all ds[variable] data are strings

    if not all(isinstance(string, str) for string in ds[variable].data):
        error = "Not all data in {} variable are strings. ".format(variable) + \
            "Data types found: {}.".format(set([type(string).__name__ for string in ds[variable].data]))

        raise ValueError(error)

    encoding = {string: number for  number, string in enumerate(np.unique(ds[variable]))}
    encoded = np.array([encoding[string] for string in ds[variable].data])
    encoded = one_hot(encoded)
    return encoded


def numeric(mylist:list) -> list[float]:
    """Return a list of floats from a list of strings."""
    def is_numeric(item):
        try:
            float(item)
            return True
        except ValueError:
            return False
    return [float(item) for item in mylist if is_numeric(item)]


def label_data(data:xr.DataArray, thresholds:list=[15, np.inf]) -> xr.DataArray:
    thresholds = sorted(thresholds)
    data_labels = 0 * data + 1 # initialise with normal climate label
    for i, lower_bound in enumerate(thresholds):
        # Returns elements from ‘DataArray’, where ‘cond’ is True, otherwise fill in ‘other’.
        data_labels = data_labels.where(data < lower_bound, int(i + 2))
    return data_labels


def weight_labels(x:np.ndarray) -> np.ndarray:
    """Return a weight for each label in x"""
    counts = Counter(x)
    weights = np.vectorize(counts.__getitem__)(x)
    weights = 1.0 / weights
    return weights * 0 + 1. # NOTE: made all ones!!


def sample_dict(data, condition="maxwind") -> dict:
        labels = data['label'].data.astype(int)
        samples = {
            'uniform': data['uniform'].data.astype(np.float32),
            'condition': data[condition].data.astype(np.float32),
            'label': labels,
            'weight': weight_labels(labels), # TODO: made even weighting to start
            'season': encode_strings(data, "season"),
            'days_since_epoch': data['time'].data.astype(int),
            }
        return samples


def check_validity(dataset, name:str) -> None:
    """Check that uniform data is in [0, 1] range.
    -- depends on ECDF or Semi-CDF data."""
    max_unif = dataset['uniform'].max()
    min_unif = dataset['uniform'].min()
    if max_unif > 1.0 or min_unif < 0.0:
        print(
            "WARNING: Uniform data in {} dataset is not in [0, 1] range. ".format(name) + 
            "Max: {:.3f}, Min: {:.3f}".format(max_unif, min_unif)
        )
    else:
        print("GOOD: Uniform data in {} dataset is in [0, 1] range.".format(name))


def make_xr_grid(ds, x='lon', y='lat', varname='x') -> xr.Dataset:
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=varname)
    h, w = ds.dims[y], ds.dims[x]
    grid = np.arange(0, h * w, 1).reshape(h, w)
    grid = xr.DataArray(
        grid, dims=[y, x], coords={y: ds[y][::-1], x: ds[x]}
    )
    ds['grid'] = grid
    return ds


def _prep_xr_data(
        datadir:str, thresholds:list=[15, np.inf], train_size=0.8, fields=['u10', 'tp'],
        epoch='1940-01-01', verbose=True, testyear=TEST_YEAR
        ) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Library-agnostic data loader for training.

    Returns:
    --------
    train : Dataset
        Train dataset with (footprint, condition, label)
    valid : Dataset
        Validation dataset with (footprint, condition, label)
    metadata : dict
        Dict with useful metadata
    """
    gc.disable() # slight speed up
    data = xr.open_dataset(os.path.join(datadir, 'data.nc')) # , engine='zarr'
    pretrain = xr.open_dataset(
        os.path.join(datadir, 'data_pretrain.nc'),
        chunks={'time': 1000}
        )
    pretrain = pretrain.transpose("time", "lat", "lon", "field")

    # make sure marginal percentiles are in (0, 1)
    check_validity(data, "training")
    check_validity(pretrain, "pretraining")

    # remove test year from both datasets
    data = data.sel(time=data['time.year'] != testyear)
    pretrain = pretrain.sel(time=pretrain['time.year'] != testyear)

    print("Pretrain shape: {}".format(pretrain['uniform'].data.shape))
    print("Train shape: {}".format(data['uniform'].data.shape))
    print("\nData loaded. Processing data...")
    
    metadata = {} # start collecting metadata

    # conditioning & sampling variables
    metadata['epoch'] = np.datetime64(epoch)
    data['maxwind'] = data.sel(field='u10')['anomaly'].max(dim=['lon', 'lat'])
    data['label'] = label_data(data['maxwind'], thresholds)
    data['season'] = data['time.season']
    data['time'] = (data['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    data = data.sel(field=fields)

    # conditioning & sampling variables (pretrain)
    pretrain['maxwind'] = pretrain.sel(field='u10')['anomaly'].max(dim=['lon', 'lat']) # anomaly
    pretrain['label'] = (0 * pretrain['maxwind']).astype(int) # zero indicates normal climate data
    pretrain['season'] = pretrain['time.season']
    pretrain['time'] = (pretrain['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    pretrain = pretrain.sel(field=fields)

    # train/test split
    if isinstance(train_size, float):
        n = data['time'].size
        train_size = int(train_size * n)

    dynamic_vars = [var for var in data.data_vars if 'time' in data[var].dims]
    static_vars = [var for var in data.data_vars if 'time' not in data[var].dims]

    train_dates = np.random.choice(data['time'].data, train_size, replace=False)
    train = xr.merge([
        data[dynamic_vars].sel(time=train_dates),
        data[static_vars]
    ])
    valid = xr.merge([
        data[dynamic_vars].sel(time=data.time[~data.time.isin(train_dates)]),
        data[static_vars]
    ])

    #  get metadata before batching and resampling
    metadata['train'] = train[['uniform', 'anomaly', 'medians', 'params']]
    metadata['valid'] = valid[['uniform', 'anomaly', 'medians', 'params']]

    # try concatenating with training data
    train = xr.concat([train, pretrain], dim='time', data_vars="minimal")
    labels = da.unique(train['label'].data).astype(int).compute().tolist()
    metadata['labels'] = labels

    # compute
    train = train.compute()
    valid = valid.compute()

    # get value counts for resampling
    metadata['train_counts'] = Counter(train['label'].data)
    metadata['valid_counts'] = Counter(valid['label'].data)

    if verbose:
        print("\nData summary:\n-------------")
        print("{:,.0f} training samples from very stormy dataset".format(metadata['train_counts'][2.0]))
        print("{:,.0f} training samples from stormy dataset".format(metadata['train_counts'][1.0]))
        print("{:,.0f} training samples from normal climate dataset".format(metadata['train_counts'][0.0]))

        print("{:,.0f} validation samples from very stormy dataset".format(metadata['valid_counts'][2.0]))
        print("{:,.0f} validation samples from stormy dataset".format(metadata['valid_counts'][1.0]))

    gc.enable()
    gc.collect()
    return train, valid, metadata


def _equal(a, b, verbose=False) -> bool:
    """Recursively check if two objects are equal.
    
    Examples:
    ---------
    >>> equal(['abc'], ['abc'])
    >>> equal(1, 1)
    >>> equal({'a': 1}, {'a': 1})
    >>> equal(True, False)
    >>> equal('abc', 'abc') 
    """
    if verbose:
        print("{} == {}?".format(a, b))

    if (a is None):
        return b is a
    
    if isinstance(a, str) and isinstance(b, str):
        return (a == b)

    if isinstance(a, dict) and isinstance(b, dict):
        if list(a.keys()) != list(b.keys()):
            return False
        return all(_equal(a[key], b[key]) for key in a.keys())
    
    if hasattr(a, '__iter__') and hasattr(b, '__iter__'):
        if len(a) != len(b):
            print(f"{a} != {b}")
            return False
        return all([_equal(i, j) for i, j in zip(a, b)])
    
    # print(f"Result {a} == {b}: {a==b}")
    return (a == b)


def _load_xr_cached(**kwargs) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Cache prepped data for faster loading."""
    if kwargs.get('cache'):
        datadir = kwargs.get('datadir')
        cachedir = os.path.join(datadir, 'cache')

        if os.path.exists(cachedir):
            # check if file containing args exists in cache dir
            kwargfile = os.path.join(cachedir, 'kwargs.npz')

            if os.path.exists(kwargfile):
                print("Loading cached arguments...")
                cached_kwargs = np.load(kwargfile, allow_pickle=True)
                cached_kwargs = {key: value[()] for key, value in cached_kwargs.items()}
                print("Current arguments: {}".format(kwargs))
                print("Cached arguments: {}".format(cached_kwargs))
                # args_match = all([kwargs.get(key) == cached_kwargs.get(key) for key in kwargs.keys()])
                args_match = _equal(kwargs, cached_kwargs)
                
                if args_match:
                    print("Arguments match cached arguments. Loading data...")
                    train = xr.open_dataset(os.path.join(cachedir, 'train.nc'))
                    valid = xr.open_dataset(os.path.join(cachedir, 'valid.nc'))
                    metadata = np.load(os.path.join(cachedir, 'metadata.npz'), allow_pickle=True)
                    metadata = {key: value[()] for key, value in metadata.items()}
                    metadata['train'] = xr.Dataset(metadata['train'])
                    metadata['valid'] = xr.Dataset(metadata['valid'])
                    return train, valid, metadata
                else:
                    print("Arguments do not match cached arguments. Remaking data.")
    # if these conditions are not met, return Nones
    return None, None, None


def _cache_xr_data(train, valid, metadata, **kwargs):
    """Cache prepped data for faster loading."""
    if kwargs.get('cache'):
        datadir = kwargs.get('datadir')
        cachedir = os.path.join(datadir, 'cache')

        # check if file containing args exists in cache dir
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)

        argfile = os.path.join(cachedir, 'kwargs.npz')

        print("Caching arguments...")
        np.savez(argfile, **kwargs)

        print("Caching data...")
        train.to_netcdf(os.path.join(cachedir, 'train.nc'))
        valid.to_netcdf(os.path.join(cachedir, 'valid.nc'))

        # metadata can only be saved to npz as DataArrays
        metadata_train = metadata['train']
        metadata_valid = metadata['valid']
        train_arrays = {var: metadata_train[var] for var in metadata_train}
        valid_arrays = {var: metadata_valid[var] for var in metadata_valid}
        metadata['train'] = train_arrays
        metadata['valid'] = valid_arrays

        np.savez(os.path.join(cachedir, 'metadata.npz'), **metadata)
        print("Data cached to {}".format(cachedir))


def load_xr_data(datadir:str, thresholds:list=[15, np.inf],
         train_size=0.8, fields=['u10', 'tp'], epoch='1940-01-01',
         verbose=True, testyear=TEST_YEAR, cache=True) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Library-agnostic data loader for training.

    Returns:
    --------
    train : Dataset
        Train dataset with (footprint, condition, label)
    valid : Dataset
        Validation dataset with (footprint, condition, label)
    metadata : dict
        Dict with useful metadata
    """
    kwargs = locals()
    if cache:
        train, valid, metadata = _load_xr_cached(**kwargs)
        if train is not None:
            # load from cache if available
            print("Data loaded from cache.")
            return train, valid, metadata
    
    # if there's no cached data, load and (optionally) cache
    train, valid, metadata = _prep_xr_data(datadir, thresholds, train_size, fields, epoch, verbose, testyear)
    _cache_xr_data(train, valid, metadata, **kwargs)
    return train, valid, metadata
