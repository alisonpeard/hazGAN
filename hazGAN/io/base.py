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
from warnings import warn
# from ..constants import TEST_YEAR
TEST_YEAR = 2021


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


def label_data(data:xr.DataArray, label_ratios:dict={'pre':1/3., 7:1/3, 20:1/3}
               ) -> xr.DataArray:
    """Apply labels to storm data using user-provided dict."""
    ratios = list(label_ratios.values())
    assert np.isclose(sum(ratios), 1), "Ratios must sum to one, received {}.".format(sum(ratios))

    labels = numeric(list(label_ratios.keys()))
    labels = sorted(labels)
    data_labels = 0 * data['maxwind'] + 1

    for label, lower_bound in enumerate(labels):
        data_labels = data_labels.where(data['maxwind'] < lower_bound, int(label + 2))

    return data_labels


def weight_labels(x:np.ndarray) -> np.ndarray:
    """Return a weight for each label in x"""
    counts = Counter(x)
    weights = np.vectorize(counts.__getitem__)(x)
    weights = 1.0 / weights
    return weights


def sample_dict(data, condition="maxwind") -> dict:
        labels = data['label'].data.astype(int)
        samples = {
            'uniform': data['uniform'].data.astype(np.float32),
            'condition': data[condition].data.astype(np.float32),
            'label': labels,
            'weight': weight_labels(labels),
            'season': encode_strings(data, "season"),
            'days_since_epoch': data['time'].data.astype(int),
            }
        return samples


def check_validity(dataset, name:str) -> None:
    """Check that uniform data is in (0, 1) range."""
    max_unif = dataset['uniform'].max()
    min_unif = dataset['uniform'].min()
    if max_unif >= 1.0 or min_unif <= 0.0:
        print(
            "WARNING: Uniform data in {} dataset is not in (0, 1) range. ".format(name) + 
            "Max: {:.3f}, Min: {:.3f}".format(max_unif, min_unif)
        )
    else:
        print("GOOD: Uniform data in {} dataset is in (0, 1) range.".format(name))

def prep_xr_data(datadir:str, label_ratios={'pre':1/3, 15: 1/3, 999:1/3},
         train_size=0.8, fields=['u10', 'tp'], epoch='1940-01-01',
         verbose=True, testyear=TEST_YEAR) -> tuple[xr.Dataset, xr.Dataset, dict]:
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
    data['label'] = label_data(data, label_ratios)
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

# %%
