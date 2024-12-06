"""
Improved input-output methods to flexibly load from pretraining and training sets.
"""
# %%
import os
import gc
import time
from typing import Union
from environs import Env
from numbers import Number
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import tensorflow as tf
from tensorflow.data import Dataset
from .constants import PADDINGS


def print_if_verbose(string:str, verbose=True) -> None:
    """Self-explanatory"""
    if verbose:
        print(string)


def encode_strings(ds:xr.Dataset, variable:str) -> tf.Tensor:
    """One-hot encode a string variable"""
    # check that all ds[variable] data are strings

    if not all(isinstance(string, str) for string in ds[variable].data):
        error = "Not all data in {} variable are strings. ".format(variable) + \
            "Data types found: {}.".format(set([type(string).__name__ for string in ds[variable].data]))

        raise ValueError(error)

    encoding = {string: number for  number, string in enumerate(np.unique(ds[variable]))}
    encoded = np.array([encoding[string] for string in ds[variable].data])
    depth = len(list(encoding.keys()))
    return tf.one_hot(encoded, depth)
    

def label_data(data, label_ratios:dict={'pre':1/3., 7:1/3, 20:1/3}) -> xr.DataArray:
    """Apply labels to storm data using user-provided dict."""
    ratios = list(label_ratios.values())
    assert np.isclose(sum(ratios), 1), "Ratios must sum to one, got {}.".format(sum(ratios))

    nlabels = len(list(label_ratios.keys())) - 1 # excluding pretrain
    labels = 0 * data['maxwind'] + nlabels       # assign highest label to everything

    for label, lower_bound in enumerate(label_ratios.keys()):
        if isinstance(lower_bound, Number): # first is always 'pre'
            labels = labels.where(data['maxwind'] < lower_bound, int(label))

    return labels


def load_data(datadir:str, condition="maxwind", label_ratios={'pre':1/3, 7: 1/3, 20:1/3},
         train_size=0.8, channels=['u10', 'tp'], image_shape=(18, 22),
         padding_mode='reflect', gumbel=True, batch_size=16,
         verbose=True) -> tuple[Dataset, Dataset, dict]:
    """Main data loader for training.

    Returns:
    --------
    train : Dataset
        Train dataset with (footprint, condition, label)
    valid : Dataset
        Validation dataset with (footprint, condition, label)
    metadata : dict
        Dict with useful metadata
    """
    assert condition in ['maxwind', 'time.season', 'label']
    gc.disable() # slight speed up

    print("\nLoading training data (hang on)...")
    start = time.time() # time data loading
    data = xr.open_dataset(os.path.join(datadir, 'data.nc')) # , engine='zarr'
    pretrain = xr.open_dataset(
        os.path.join(datadir, 'data_pretrain.nc'),
        chunks={'time': 1000}
        )
    pretrain = pretrain.transpose("time", "lat", "lon", "channel")

    print("Pretrain shape: {}".format(pretrain['uniform'].data.shape))
    print("Train shape: {}".format(data['uniform'].data.shape))
    print("\nData loaded. Processing data...\n")
    
    metadata = {} # start collecting metadata

    # conditioning & sampling variables
    metadata['epoch'] = np.datetime64('1950-01-01')
    data['maxwind'] = data.sel(channel='u10')['anomaly'].max(dim=['lon', 'lat'])
    data['label'] = label_data(data, label_ratios)
    data['season'] = data['time.season']
    data['time'] = (data['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    data = data.sel(channel=channels)

    # conditioning & sampling variables (pretrain)
    pretrain['maxwind'] = pretrain.sel(channel='u10')['anomaly'].max(dim=['lon', 'lat']) # anomaly
    pretrain['label'] = (0 * pretrain['maxwind']).astype(int) # zero indicates normal climate data
    pretrain['season'] = pretrain['time.season']
    pretrain['time'] = (pretrain['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    pretrain = pretrain.sel(channel=channels)

    if verbose:
        print("\nData summary:\n-------------")
        print("{:,.0f} samples from storm dataset".format(data.sizes['time']))
        print("{:,.0f} samples from normal climate dataset".format(pretrain.sizes['time']))

    # train/test split
    if isinstance(train_size, float):
        n = data['time'].size
        train_size = int(train_size * n)

    dynamic_vars = [var for var in data.data_vars if 'time' in data[var].dims]
    static_vars = [var for var in data.data_vars if 'time' not in data[var].dims]

    train_dates = np.random.choice(data['time'].data, train_size, replace=False)
    train= xr.merge([
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
    metadata['paddings'] = PADDINGS()

    def sample_dict(data):
        samples = {
            'uniform': data['uniform'].data.astype(np.float32),
            'condition': data[condition].data.astype(np.float32),
            'label': data['label'].data.astype(int),
            'season': encode_strings(data, "season"),
            'days_since_epoch': data['time']
            }
        return samples

    train = Dataset.from_tensor_slices(sample_dict(train)).shuffle(10_000)
    valid = Dataset.from_tensor_slices(sample_dict(valid)).shuffle(500)

    # manual under/oversampling
    split_train = [train.filter(lambda sample: sample['label']==label) for label in labels]
    target_dist = list(label_ratios.values())
    train = tf.data.Dataset.sample_from_datasets(split_train, target_dist)

    #  Define transformations
    def gumbel(uniform, eps=1e-6):
        tf.debugging.Assert(tf.less_equal(tf.reduce_max(uniform), 1.), [uniform])
        tf.debugging.Assert(tf.greater_equal(tf.reduce_min(uniform), 0.), [uniform])
        uniform = tf.clip_by_value(uniform, eps, 1-eps)
        return -tf.math.log(-tf.math.log(uniform))

    def transforms(sample):
        uniform = sample['uniform']
        uniform = tf.image.resize(uniform, image_shape)
        if gumbel:
            uniform = gumbel(uniform)
        if padding_mode is not None:
            paddings = PADDINGS()
            uniform = tf.pad(uniform, paddings, mode=padding_mode)
        sample['uniform'] = uniform
        return sample

    train = train.map(transforms)
    valid = valid.map(transforms)

    # pipeline methods
    train = train.shuffle(10_000)
    train = train.repeat()
    train = train.batch(batch_size)

    valid = valid.batch(batch_size, drop_remainder=True)

    end = time.time()
    print('\nTime taken to load datasets: {:.2f} seconds.\n'.format(end - start))
    gc.enable()
    gc.collect()
    return train, valid, metadata


# %%
if __name__ == "__main__":
    print('Testing io.py...')
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    train, valid, metadata = load_data(datadir)

    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for i in range(10):
                batch = next(iter(dataset))
        print("10 epoch execution time:", time.perf_counter() - start_time)

    benchmark(train)
    benchmark(valid)

    print("param shapes: {}".format(metadata['train']['params']))

    # train.save(os.path.join(datadir, 'train_dataset'))
    # valid.save(os.path.join(datadir, 'valid_dataset'))
# %%
# TODO: Make a class to handle train, valid, and metadata
if False:
    import json
    class MetaDataset(Dataset):
        
        def __init__(self, datadir:str, condition="maxwind",
                    label_ratios={'pre':1/3, 7: 1/3, 20:1/3},
                    train_size=0.8, channels=['u10', 'tp'],
                    image_shape=(18, 22), padding_mode='reflect',
                    gumbel=True, batch_size=16, **kwargs):
            super().__init__(**kwargs)
            self.datasets = []
            self.datadir = datadir
            self.condition = condition
            self.label_ratios = label_ratios,
            self.train_size = train_size
            self.channels = channels
            self.image_shape = image_shape
            self.padding_mode = padding_mode
            self.gumbel = gumbel
            self.batch_size = {}

            self.kwargs = {}
            self.metadata = {}

        def __len__():
            pass

        def load(self):
            self.train = super().load(os.path.join(datadir, 'train'))
            self.valid = super().load(os.path.join(datadir, 'valid'))
            # self.metadata = 

        def save(self):
            self.train.save(os.path.join(self.datadir, "train"))
            self.train.save(os.path.join(self.datadir, "valid"))
            with open(os.path.join(self.dir, 'metadata.json'), 'wb') as fp:
                json.dumps(self.metadata, fp)



