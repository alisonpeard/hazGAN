"""Data handling methods for the hazGAN model."""
import os
import tensorflow as tf
import xarray as xr
from .extreme_value_theory import gumbel
from .utils import KNOWN_OUTLIERS

def load_datasets(datadir, ntrain, padding_mode='reflect', image_shape=(18, 22), gumbel_marginals=False, batch_size=32):
    [train_u, test_u], *_ = load_training(datadir, ntrain, padding_mode, image_shape, gumbel_marginals=gumbel_marginals)
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(batch_size)
    return train, test


def load_training(datadir, ntrain, padding_mode='constant', image_shape=(18, 22),
                  numpy=False, gumbel_marginals=True, channels=['u10', 'tp'],
                  uniform='uniform', u10_min=-999.):
    """
    Load the hazGAN training data from the data.nc file.

    Parameters:
    ----------
    datadir : str
        Directory where the data.nc file is stored.
    ntrain : int
        Number of training samples.
    padding_mode : {'constant', 'reflect', 'symmetric', None}, default 'constant'
        Padding mode for the uniform-transformed marginals.
    image_shape : tuple, default=(18, 22)
        Shape of the image data.
    numpy : bool, default False
        Whether to return numpy arrays or tensors.
    gumbel_marginals : bool, default False
        Whether to use Gumbel-transformed marginals.
    channels : list, default ['u10', 'tp']
        List of channels to use.
    uniform : {'uniform', 'uniform'}, default 'uniform'
        What type of uniform-transformed marginals to use. 'uniform' comes from
        the ECDF and 'uniform_semi' comes from the semiparametric CDF of Heffernan
        and Tawn  (2004).
    """
    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    time_no_outlier = data.where(~data.time.isin(KNOWN_OUTLIERS), drop=True).time
    data = data.sel(time=time_no_outlier)
    data = data.sel(channel=channels)

    print('Only taking footprints with max u10 anomaly greater than', u10_min)
    data['maxima'] = data.sel(channel='u10').anomaly.max(dim=['lat', 'lon'])
    time_subset = data.where(data.maxima >= u10_min, drop=True).time
    data = data.sel(time=time_subset)
    print('Number of remaining footprints:', len(data.time))
    print('Number of training samples:', ntrain)


    X = tf.image.resize(data.anomaly, image_shape)
    U = tf.image.resize(data[uniform], image_shape)
    M = tf.image.resize(data.medians, image_shape)
    z = data.storm_rp.values
    params = data.params.values
    
    if padding_mode is not None:
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        U = tf.pad(U, paddings, mode=padding_mode)

    # training is on most recent data
    train_u = U[-ntrain:, ...]
    test_u = U[:-ntrain, ...]
    train_x = X[-ntrain:, ...]
    test_x = X[:-ntrain, ...]
    train_m = M[-ntrain:, ...]
    test_m = M[:-ntrain, ...]
    train_z = z[-ntrain:]
    test_z = z[:-ntrain]

    if gumbel_marginals:
        train_u = gumbel(train_u)
        test_u = gumbel(test_u)

    if numpy:
        train_u = train_u.numpy()
        test_u = test_u.numpy()
        train_x = train_x.numpy()
        test_x = test_x.numpy()
        train_m = train_m.numpy()
        test_m = test_m.numpy()

    # replace nans with zeros
    train_u = tf.where(tf.math.is_nan(train_u), tf.zeros_like(train_u), train_u)
    test_u = tf.where(tf.math.is_nan(test_u), tf.zeros_like(test_u), test_u)
    train_mask = tf.where(tf.math.is_nan(train_u))
    test_mask = tf.where(tf.math.is_nan(test_u))

    # return a dictionary to keep it tidy
    training = {'train_u': train_u, 'test_u': test_u, 'train_x': train_x, 'test_x': test_x,
                'train_m': train_m, 'test_m': test_m, 'train_z': train_z, 'test_z': test_z,
                'params': params, 'train_mask': train_mask, 'test_mask': test_mask}
    
    return training

