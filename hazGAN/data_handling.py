import os
import tensorflow as tf
import xarray as xr
from sklearn.preprocessing import StandardScaler
from .extreme_value_theory import gumbel


def load_datasets(datadir, ntrain, padding_mode='constant', image_shape=(21, 21), gumbel_marginals=False, batch_size=32):
    [train_u, test_u], *_ = load_training(datadir, ntrain, padding_mode, image_shape, gumbel_marginals=gumbel_marginals)
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(batch_size)
    return train, test


def load_training(datadir, ntrain, padding_mode='constant', image_shape=(21, 21), numpy=False, gumbel_marginals=False):
    """Note numpy arrays will appear upside down because of latitude."""
    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    X = tf.image.resize(data.anomaly, image_shape)
    U = tf.image.resize(data.uniform, image_shape)
    M = tf.image.resize(data.medians, image_shape)
    z = data.storm_rp.values
    params = data.params.values
    
    if padding_mode is not None:
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        U = tf.pad(U, paddings, mode=padding_mode)

    train_u = U[:ntrain, ...]
    test_u = U[ntrain:, ...]
    train_x = X[:ntrain, ...]
    test_x = X[ntrain:, ...]
    train_m = M[:ntrain, ...]
    test_m = M[ntrain:, ...]
    train_z = z[:ntrain]
    test_z = z[ntrain:]

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

    return [train_u, test_u], [train_x, test_x], [train_m, test_m], [train_z, test_z], params, train_mask, test_mask

