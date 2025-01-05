# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import warnings
from typing import Tuple, Union


def tuple_to_numpy(tup):
    tup = tuple([x.item() for x in tup])
    return tup


def tuple_to_torch(tup):
    if isinstance(tup[0], torch.Tensor):
        return tup
    tup = tuple([torch.tensor(x) for x in tup])
    return tup


def downsize(insize, k, s, p) -> tuple:
    h = torch.floor(((insize[0]-k[0]+2*p)/s) + 1).int()
    w = torch.floor(((insize[1]-k[1]+2*p)/s) + 1).int()
    return (h, w)


def upsize(insize, k, s, p) -> tuple:
   h = torch.floor((insize[0] - 1) * s - 2 * p + (k[0] - 1) + 1).int()
   w = torch.floor((insize[1] - 1) * s - 2 * p + (k[1] - 1) + 1).int()
   return (h, w)
   

def upsample(x, kernel_size:tuple, stride:int=1, padding:int=0, mode='bilinear') -> torch.Tensor:
    """Upsample tensor x using interpolation."""
    insize = x.size()[2:]
    kernel_size = tuple_to_torch(kernel_size)
    outsize = upsize(insize, kernel_size, stride, padding)
    upsampled = F.interpolate(x, size=outsize, mode=mode)
    return upsampled


def downsample(x, kernel_size:tuple, stride:int=1, padding:int=0) -> torch.Tensor:
    """Downsample tensor x using average pooling."""
    kernel_size = tuple_to_torch(kernel_size)
    downsampled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    return downsampled



class GumbelBlock(nn.Module):
    """Total experiment, NOTE: not using."""
    def __init__(self, num_features, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.make_kind_of_uniform = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.make_kind_of_uniform(x)
        x = -torch.log(-torch.log(x + self.epsilon) + self.epsilon)
        return x


class ResidualUpBlock(nn.Module):
    """Single residual block for upsampling (increasing resolution)."""
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:Tuple,
                 stride:int=1,
                 padding:int=0,
                 upsample_mode:str='bilinear',
                 dropout:Union[None, float]=None,
                 noise_sd:Union[None, float]=None, 
                 **kwargs
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.upsample = lambda x: upsample(x, kernel_size, stride, padding, upsample_mode)
        self.project = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)

        # regularisation attributes
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
        self.noise_sd = noise_sd
    

    def regularise(self, x):
        x = self.dropout(x)
        if self.training and self.noise_sd is not None:
            noise = torch.randn_like(x) * self.noise_sd
            x = x + noise
        return x

    
    def forward(self, x) -> torch.Tensor:
        identity = self.upsample(x)
        identity = self.project(identity)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.regularise(x)
        return x + identity


class ResidualDownBlock(nn.Module):
    """Single residual block for downsampling (decreasing resolution)."""
    def __init__(self, in_channels:int, out_channels: int,
                 kernel_size:Union[int, Tuple[int, int]],
                 stride:Union[int, Tuple[int, int]]=1,
                 padding:Union[int, Tuple[int, int]]=0,
                 dropout:Union[None, float]=None,
                 noise_sd:Union[None, float]=None, 
                 **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU() # GELU()
        self.downsample = lambda x: downsample(x, kernel_size, stride, padding)
        self.project = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)

        # regularisation attributes
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
        self.noise_sd = noise_sd

    def regularise(self, x:torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        if self.training and self.noise_sd is not None:
            noise = torch.randn_like(x) * self.noise_sd
            x = x + noise
        return x
    

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = self.project(self.downsample(x))
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.regularise(x)
        return x + identity


# %% tests
if __name__ == "__main__":
    print('Running tests...')
    x = np.arange(0, 25, 1).reshape((1,1,5,5))
    x = torch.Tensor(x)
    xsize = x.shape[2:]

    kernel_sizes = [(2, 2), (2, 3)]
    strides = [1, 2]
    paddings = [0, 1]

    for kernel_size in kernel_sizes:
        for stride in strides:
            for padding in paddings:
                # test upsampling
                kernel_size = tuple_to_torch(kernel_size)
                up_expected = upsize(xsize, kernel_size, stride, padding)
                up_expected = tuple_to_numpy(up_expected)
                up_calculated = upsample(x, kernel_size, stride, padding).size()[2:]
                up_calculated = tuple(up_calculated)
                assert up_calculated == up_expected

                # # test downsampling
                down_expected = downsize(xsize, kernel_size, stride, padding)
                down_expected = tuple_to_numpy(down_expected)
                down_calculated = downsample(x, kernel_size, stride, padding).size()[2:]
                down_calculated = tuple(down_calculated)
                assert down_calculated == down_expected

                # # test upblock
                upblock = ResidualUpBlock(1, 2, kernel_size, stride, padding)
                y = upblock(x)
                y2 = nn.ConvTranspose2d(1, 2, kernel_size, stride, padding)(x)
                assert y.shape == y2.shape

                # test downblock
                downblock = ResidualDownBlock(1, 2, kernel_size, stride, padding)
                y = downblock(x)
                y2 = nn.Conv2d(1, 2, kernel_size, stride, padding)(x)
                assert y.shape == y2.shape


    print('All tests passed!')


# %%