"""
Improved input-output methods to flexibly load from pretraining and training sets.
"""
# %%
import os
import time
from typing import Union
from environs import Env
from numbers import Number
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.data import Dataset


PADDINGS = tf.constant([[1, 1], [1, 1], [0, 0]])


def print_if_verbose(string:str, verbose=True) -> None:
    """Self-explanatory"""
    if verbose:
        print(string)


def process_outliers(datadir, verbose=True) -> Union[list[np.datetime64], None]:
    """Identify wind bombs using Frobernius inner product (find code)"""
    outlier_file = os.path.join(datadir, 'outliers.csv')
    if os.path.isfile(outlier_file):
        print_if_verbose("Loading file containing outliers...", verbose)
        outliers = pd.read_csv(outlier_file, index_col=[0])
        outliers = pd.to_datetime(outliers['time']).to_list()
        outliers = [np.datetime64(date) for date in outliers]
        return outliers
    else:
        print_if_verbose("No outlier file found.", verbose)
        return None
    

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
    labels = 0 * data['maxwind'] + nlabels

    for label, lower_bound in enumerate(label_ratios.keys()):
        if isinstance(lower_bound, Number): # first is always 'pre'
            labels = labels.where(data['maxwind'] < lower_bound, int(label))

    return labels


def load_data(datadir:str, condition="maxwind", label_ratios={'pre':1/3, 7: 1/3, 20:1/3},
         train_size=0.8, channels=['u10', 'tp'], image_shape=(18, 22),
         padding_mode='reflect', gumbel=True, batch_size=16) -> tuple[Dataset, Dataset, dict]:
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

    print("\nLoading training data (hang on) ...")
    start = time.time() # time data loading
    data = xr.open_dataset(os.path.join(datadir, 'data.nc'))
    pretrain = xr.open_dataset(os.path.join(datadir, 'data_pretrain.nc'))

    def select_no_broadcast(data, selection):
        """Too hardcoded to put outside function."""
        dynamic_vars = [var for var in data.data_vars if 'time' in data[var].dims]
        static_vars = [var for var in data.data_vars if 'time' not in data[var].dims]
        data = xr.merge([
            data[dynamic_vars].sel(time=selection),
            data[static_vars]
        ])
        return data

    outliers = process_outliers(datadir)
    if outliers is not None:
        data = select_no_broadcast(data, data.time[~data.time.isin(outliers)])
        pretrain = pretrain.where(~pretrain['time'].isin(outliers), drop=True)
    
    metadata = {} # start collecting metadata

    # conditioning & sampling variables
    metadata['epoch'] = np.datetime64('1950-01-01')
    data['maxwind'] = data.sel(channel='u10')['anomaly'].max(dim=['lon', 'lat']) # anomaly
    data['label'] = label_data(data, label_ratios)
    data['season'] = data['time.season']
    data['time'] = (data['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    data = data.sel(channel=channels)

    # conditioning & sampling variables (pretrain)
    pretrain['maxwind'] = pretrain.sel(channel='u10')['anomaly'].max(dim=['lon', 'lat']) # anomaly
    pretrain['label'] = (0 * pretrain['maxwind']).astype(int) # zero marks training data
    pretrain['season'] = pretrain['time.season']
    pretrain['time'] = (pretrain['time'].values - metadata['epoch']).astype('timedelta64[D]').astype(np.int64)
    pretrain = pretrain.sel(channel=channels)

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
    labels = list(np.unique(train['label'].data).astype(int))
    metadata['labels'] = labels
    metadata['paddings'] = PADDINGS

    def sample_dict(data):
        samples = {
            'uniform': data['uniform'].data.astype(np.float32),
            'condition': data[condition].data.astype(np.float32),
            'label': data['label'].data.astype(int),
            'season': encode_strings(data, "season"),
            'days_since_epoch': data['time']
            }
        return samples

    train = Dataset.from_tensor_slices(sample_dict(train)).shuffle(100)
    valid = Dataset.from_tensor_slices(sample_dict(valid)).shuffle(100)

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
            paddings = PADDINGS
            uniform = tf.pad(uniform, paddings, mode=padding_mode)
        sample['uniform'] = uniform
        return sample

    train = train.map(transforms)
    valid = valid.map(transforms).batch(batch_size, drop_remainder=True)

    # manual under/oversampling
    split_train = [train.filter(lambda sample: sample['label']==label) for label in labels]
    target_dist = list(label_ratios.values())

    train = tf.data.Dataset.sample_from_datasets(split_train, target_dist).batch(batch_size)
    end = time.time()
    print('Time taken to load datasets: {:.2f} seconds.\n'.format(end - start))

    train = train.prefetch(tf.data.AUTOTUNE)
    valid = valid.prefetch(tf.data.AUTOTUNE)
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



