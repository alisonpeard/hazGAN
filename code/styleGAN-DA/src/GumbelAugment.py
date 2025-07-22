# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# modify these to work with data on [0, 1)

import torch
import torch.nn.functional as F
from functools import partial

EPS = 1e-6

def GumbelAugment(x, nimg, total_kimg, p=0.8, channels_first=True):
    """
    TODO: make p adaptive?
    """
    if not channels_first:
        x = x.permute(0, 3, 1, 2) # BCHW
    progress = min(nimg / (1000 * total_kimg), 1.0)
    x = gumbel_transform(x, progress, p)
    if not channels_first:
        x = x.permute(0, 2, 3, 1)
    x = x.contiguous()
    return x


def gumbel_transform(x, p=0.8, eps=1e-6):
    """Gumbel transform random selection of points and scale to keep gradients chill.
    """
    def optimal_K(x):
        with torch.no_grad():
            sample_points = x
            if len(sample_points) > 0:
                gumbel_grads = 1 / (sample_points * torch.log(sample_points))
                K = 1 / torch.sqrt(torch.mean(gumbel_grads ** 2))
            else:
                K = 0.1 
        return K

    def gumbel_transform(x):
        x = torch.clamp(x, eps, 1 - eps)
        K = optimal_K(x)
        # K = K + (1 - K) * progress
        x = -torch.log(-torch.log(x))
        return K * x
    
    f = partial(gumbel_transform, progress=progress)
    threshold = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = torch.where(threshold < p, f(x), x)

    return x
