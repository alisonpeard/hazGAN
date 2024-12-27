# %%
# untested
import gc
import time
from environs import Env
import tensorflow as tf
from tensorflow.data import Dataset
from ..constants import TEST_YEAR, PADDINGS
from ..data import prep_xr_data, sample_dict

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
    train, valid, metadata = prep_xr_data(datadir, label_ratios, train_size, fields, verbose, testyear)
    labels = metadata["labels"]

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



