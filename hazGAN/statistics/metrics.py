# %%
import numpy as np
from .base import *


def chi_rmse(real, fake):
    c = real.shape[1]
    chi_diff = np.zeros(c)
    for channel in range(c):
        real_channel = real[:, channel, :, :]
        fake_channel = fake[:, channel, :, :]
        chi_diff[channel] = chi_diff_1d(real_channel, fake_channel)
    return np.mean(chi_diff)


def chi_diff_1d(real, fake):
    ecs_real = pairwise_extremal_coeffs(real)
    ecs_fake = pairwise_extremal_coeffs(fake)
    diff = ecs_real - ecs_fake
    return np.sqrt(np.mean(np.square(diff)))


def pairwise_extremal_coeffs(uniform):
    """Calculate extremal coefficients for each pair of pixels across single channel."""
    assert (
        len(uniform.shape) == 3
            ), "Function all_extremal_coeffs fors for 3-dimensional tensors only."
    n, h, w = uniform.shape
    uniform = np.reshape(uniform, (n, h * w))
    frechet = inverted_frechet(uniform)
    print("frechet shape:", frechet.shape)
    minima = minner_product(frechet.T, frechet)
    n = float(n)
    minima = minima.astype(float)
    ecs = np.zeros_like(minima)
    ecs = np.divide(n, minima, out=ecs, where=minima != 0, dtype=float)
    return ecs


def minner_product(a, b, batch_size=100):
    """Use broadcasting with batching to get sum of pairwise minima."""
    result = np.zeros((4096, 4096))
    
    for i in range(0, 4096, batch_size):
        batch_end = min(i + batch_size, 4096)
        batch_a = a[i:batch_end]  # Shape: (batch_size, 1248)
        
        batch_result = np.sum(
            np.minimum(
                np.expand_dims(batch_a, axis=-1),      # Shape: (batch_size, 1248, 1)
                np.expand_dims(b, axis=0)              # Shape: (1, 1248, 4096)
            ),
            axis=1
        )
        result[i:batch_end, :] = batch_result
    
    return result


def test_minner_product():
    x = np.array([[1, 2], [1, 1]])
    assert np.array_equal(minner_product(x.T, x), np.array([[2, 2], [2, 3]])), f"{minner_product(x.T, x)}"


def get_extremal_coeffs_nd(marginals, sample_inds):
    """Calculate extremal coefficients across K-dimensional uniform data."""
    n, h, w, k = marginals.shape
    data = marginals.reshape(n, h * w, k)
    data = data[:, sample_inds, :]
    frechet = inverted_frechet(data)
    ecs = {}
    for i in range(len(sample_inds)):
        ecs[sample_inds[i]] = raw_extremal_coeff_nd(frechet[:, i, :])
    return ecs


def raw_extremal_coeff_nd(frechets: np.ndarray) -> float:
    n, d = frechets.shape
    minima = np.min(frechets, axis=1)  # minimum for each row
    minima = np.sum(minima)
    if minima > 0:
        theta = n / minima
    else:
        print("Warning: all zeros in minima array.")
        theta = d
    return float(theta)

