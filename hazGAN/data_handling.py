import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from .extreme_value_theory import gumbel


def load_datasets(datadir, ntrain, padding_mode='constant', image_shape=(18, 22), gumbel_marginals=False, batch_size=32):
    [train_u, test_u], *_ = load_training(datadir, ntrain, padding_mode, image_shape, gumbel_marginals=gumbel_marginals)
    train = tf.data.Dataset.from_tensor_slices(train_u).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_u).batch(batch_size)
    return train, test


def load_training(datadir, ntrain, padding_mode='constant', image_shape=(18, 22), numpy=False, gumbel_marginals=False):
    data = np.load(os.path.join(datadir, "data.npz"))
    X = tf.image.resize(data["X"], image_shape)
    U = tf.image.resize(data["U"], image_shape)
    M = tf.image.resize(data["M"], image_shape)
    z = data["z"]
    params = data["params"]
    
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
        min_u, max_u = tf.math.reduce_min(train_u), tf.math.reduce_max(train_u)

    if numpy:
        train_u = train_u.numpy()
        test_u = test_u.numpy()
        train_x = train_x.numpy()
        test_x = test_x.numpy()
        train_m = train_m.numpy()
        test_m = test_m.numpy()

    return [train_u, test_u], [train_x, test_x], [train_m, test_m], [train_z, test_z], params



