# %%
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def tuple_to_numpy(tup):
    tup = tuple([x.item() for x in tup])
    return tup


def tuple_to_torch(tup):
    if isinstance(tup[0], torch.Tensor):
        return tup
    tup = tuple([torch.tensor(x) for x in tup])
    return tup


def downsize(insize:tuple, k:tuple, s:int, p:int) -> tuple:
    h = int(((insize[0]-k[0]+2*p)/s) + 1)
    w = int(((insize[1]-k[1]+2*p)/s) + 1)
    return (h, w)


def upsize(insize:tuple, k:tuple, s:int, p:int) -> tuple:
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


class injectNoise(nn.Module):
    def __init__(self, channels):
        super(injectNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


class ResidualUpBlock(nn.Module):
    """Single residual block for upsampling (increasing resolution)."""
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:Tuple,
                 stride:int=1,
                 padding:int=0,
                 lrelu:float=0.2,
                 upsample_mode:str='bilinear',
                 dropout:Union[None, float]=None,
                 noise:Union[None, float]=None, 
                 **kwargs
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(lrelu)
        self.upsample = lambda x: upsample(x, kernel_size, stride, padding, upsample_mode)
        self.project = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)

        # regularisation attributes
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
        self.noise   = injectNoise(out_channels) if noise else nn.Identity()
    

    def regularise(self, x):
        x = self.dropout(x)
        x = self.noise(x)
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
    """Single residual block for downsampling (decreasing resolution).
    
    Lazy input shape initialisation for Layer norm.
    """
    def __init__(self, in_channels:int, out_channels: int,
                 kernel_size:Union[int, Tuple[int, int]],
                 stride:Union[int, Tuple[int, int]]=1,
                 padding:Union[int, Tuple[int, int]]=0,
                 lrelu:float=0.2,
                 dropout:Union[None, float]=None,
                 noise:Union[None, float]=None, 
                 **kwargs) -> None:
        super().__init__()
        # attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple_to_torch(kernel_size)
        self.stride = stride
        self.padding = padding

        # layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = None
        self._init_hook_handle = self.register_forward_pre_hook(self._initialise_layer_norm)
        self.activation = nn.LeakyReLU(lrelu)
        self.downsample = lambda x: downsample(x, kernel_size, stride, padding)
        self.project = nn.Conv2d(self.in_channels, self.out_channels, 1, 1)

        # regularisation layers
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else nn.Identity()
        self.noise   = injectNoise(out_channels) if noise else nn.Identity()


    def _initialise_layer_norm(self, module, x):
        if self.norm is None:
            input_size = x[0].size()[2:]
            output_size = downsize(input_size, self.kernel_size, self.stride, self.padding)
            output_size = torch.Size(output_size)
            # layer norm doesn't inherit device, need to explictly set
            self.norm = nn.LayerNorm(output_size).to(x[0].device)
            self._init_hook_handle.remove()


    def regularise(self, x:torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.noise(x)
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