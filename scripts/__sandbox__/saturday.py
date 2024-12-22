# %%
%load_ext autoreload
%autoreload 2  
# %%
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from environs import Env
from hazGAN.data import prep_xr_data
from hazGAN.data import sample_dict, ToTensor, Gumbel, Resize, Pad, StormDataset



# %%
if __name__ == "__main__":
    print('Testing io.py...')
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    traindata, validdata, metadata = prep_xr_data(datadir)
    traindata['uniform'].max()

    # %%make datasets
    transform = transforms.Compose([ToTensor(), Gumbel(), Resize((18, 22)), Pad('reflect', (0, 0, 2, 2))])
    train = StormDataset(sample_dict(traindata), transform=transform)
    valid = StormDataset(sample_dict(validdata), transform=transform)

    # %% make loaders
    trainsampler = WeightedRandomSampler(train.data['weight'], len(train), replacement=True)
    trainloader = DataLoader(train, batch_size=16, pin_memory=True, sampler=trainsampler)
    validloader = DataLoader(valid, batch_size=16, shuffle=False, pin_memory=True)

    # %% 
    next(iter(trainloader))['uniform'].shape
    # %%
    # %%transform wrapper
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# %% Transforms dev
    from hazGAN.torch import load_data
# %%
    train, valid, metadata = load_data(datadir, 64, "reflect", (18, 22))