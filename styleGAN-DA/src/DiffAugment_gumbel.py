# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# modify these to work with data on [0, 1)

import torch
import torch.nn.functional as F

EPS = 1e-6

def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + 0.1 * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    x = torch.clamp(x, 0, 1 - EPS)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.2) + x_mean
    x = torch.clamp(x, 0, 1 - EPS)
    return x


def rand_contrast(x):
    # scale contrast by factor of 0.9 -> 1.2
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.2 + 0.9) + x_mean
    x = torch.clamp(x, 0, 1 - EPS)
    return x

def rand_noise(x):
    # like measurement errors
    x_noise = 0.05 * torch.randn_like(x, dtype=x.dtype, device=x.device)
    x = x + x_noise
    x = torch.clamp(x, 0, 1 - EPS)
    return x


def gumbel_transform(x, p=0.255, eps=1e-6):
    """Gumbel transform random selection of points and scale to keep gradients chill.
    
    TODO: make p adaptive
    """
    def optimal_K(x):
        with torch.no_grad():
            # sample_mask = torch.rand_like(x) < 0.1
            # sample_points = x[sample_mask]
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
        x = -torch.log(-torch.log(x))
        return K * x
    
    threshold = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = torch.where(threshold < p, gumbel_transform(x), x)
    return x


# def rand_translation(x, ratio=0.125):
#     shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
#     translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
#     translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(x.size(2), dtype=torch.long, device=x.device),
#         torch.arange(x.size(3), dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
#     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
#     x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
#     x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
#     return x


# def rand_cutout(x, ratio=0.5):
#     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
#     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
#     grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
#     mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
#     mask[grid_batch, grid_x, grid_y] = 0
#     x = x * mask.unsqueeze(1)
#     return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    # 'translation': [rand_translation],
    # 'cutout': [rand_cutout],
    'climate': [rand_noise, rand_brightness, rand_saturation, rand_contrast],
    'extreme': [gumbel_transform],
}
