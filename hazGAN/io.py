"""
Improved input-output methods to flexibly load from pretraining and training sets.
"""
# %%
import os
import gc
import time
import torch
from environs import Env
import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
import dask.array as da
import torch
from collections import Counter
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data import WeightedRandomSampler

from constants import TEST_YEAR


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


def gumbel(uniform:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    assert torch.all(uniform < 1.0), "Uniform values must be < 1"
    assert torch.all(uniform > 0.0), "Uniform values must be > 0"
    uniform = torch.clamp(uniform, eps, 1-eps)
    return -torch.log(-torch.log(uniform))


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
    data = xr.open_dataset(os.path.join(datadir, 'data.nc')) # , engine='zarr'
    pretrain = xr.open_dataset(
        os.path.join(datadir, 'data_pretrain.nc'),
        chunks={'time': 1000}
        )
    pretrain = pretrain.transpose("time", "lat", "lon", "field")

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

    return train, valid, metadata

# %%
class DictDataset(Dataset):
    def __init__(self, data_dict:Dict[str, np.ndarray]):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.length = len(data_dict[self.keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch = {}
        for key in self.keys:
            batch[key] = self.data[key][idx]
        return batch
    
# %%






# %% DEV // DEBUGGING BELOW HERE ##############################################
def test_sampling_ratios(loader):
        sample = next(iter(loader))['label'].numpy()
        labels = Counter(sample)

        for sample in loader:
            labels += Counter(sample['label'].numpy())

        return labels
# %%
if __name__ == "__main__":
    print('Testing io.py...')
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    train, valid, metadata = prep_xr_data(datadir)

    # make datasets
    train = DictDataset(sample_dict(train))
    valid = DictDataset(sample_dict(valid))

    # make loaders
    train_sampler = WeightedRandomSampler(train.data['weights'], len(train), replacement=True)
    train_loader = DataLoader(train, batch_size=16, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(valid, batch_size=16, shuffle=False, pin_memory=True)

    # %%
    # %%transform wrapper
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# %% VIZ

# %%
xds = DictDataset(sample_dict(train))
sample = next(iter(xds))



# %%


def create_dataloaders(
    sample_dict: Dict[str, np.ndarray],
    labels: List[Any],
    label_ratios: Dict[Any, float],
    batch_size: int,
    image_shape: Tuple[int, int],
    padding_mode: str = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    
    # Create datasets
    dataset = CustomDataset(sample_dict)
    
    # Split into train/valid
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    
    # Split train by labels
    label_datasets = []
    data_sizes = {}

    return train_dataset
    print("\nCalculating input class sizes...")
    for label in labels:
        label_indices = [i for i in range(len(train_dataset)) 
                        if train_dataset[i]['label'] == label]
        label_datasets.append(Subset(train_dataset, label_indices))
        data_sizes[label] = len(label_indices)
    
    print("\nClass sizes:\n------------")
    for label, size in data_sizes.items():
        print(f"Label: {label} | size: {size:,}")
    
    # Create resampled dataset
    target_dist = list(label_ratios.values())
    train_dataset = ResampledDataset(label_datasets, target_dist)
    
    # Create transform
    transform = TransformWrapper(
        image_shape=image_shape,
        use_gumbel=True,
        padding_mode=padding_mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    gc.enable()
    gc.collect()
    
    return train_loader, valid_loader



# %%
def load_data(datadir:str, condition="maxwind", label_ratios={'pre':1/3, 15: 1/3, 999:1/3},
         train_size=0.8, fields=['u10', 'tp'], image_shape=(18, 22),
         padding_mode='reflect', gumbel=True, batch_size=16,
         verbose=True, testyear=TEST_YEAR) -> tuple:
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

    # process xarray datasets
    train, valid, metadata = prep_data(datadir, label_ratios, train_size, fields, verbose, testyear)

    # create dataloaders
    train_loader = create_dataloaders(
        sample_dict(train), metadata['labels'], label_ratios, batch_size, image_shape, padding_mode
    )

    end = time.time()
    print('\nTime taken to load datasets: {:.2f} seconds.\n'.format(end - start))
    gc.enable()
    gc.collect()

    return train_loader

    # below here is library-specific and old

    train = Dataset.from_tensor_slices(sample_dict(train)).shuffle(10_000)
    valid = Dataset.from_tensor_slices(sample_dict(valid)).shuffle(500)

    # manual under/oversampling
    split_train = [train.filter(lambda sample: sample['label']==label) for label in labels]

    # print size of each dataset in split_train
    print("\nCalculating input class sizes...")
    data_sizes = {}
    for label, dataset in zip(labels, split_train):
        data_sizes[label] = dataset.reduce(0, lambda x, _: x + 1).numpy()
    print("\nClass sizes:\n------------")
    for label, size in data_sizes.items():
        print("Label: {} | size: {:,.0f}".format(label, size))

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


# %% ##########################################################################
# SCROLL DOWN FOR TESTING // DEBUGGING

# %% DEV // DEBUGGING BELOW HERE ##############################################
if __name__ == "__main__":
    print('Testing io.py...')
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    train = load_data(datadir)



    # %%
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



