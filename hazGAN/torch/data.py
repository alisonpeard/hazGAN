import time
import numpy as np
from collections import Counter
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ..data import prep_xr_data, sample_dict


# Transforms
class Gumbel(object):
    """Convert uniform data to Gumbel using PIT."""
    def __call__(self, sample:dict, eps:float=1e-8) -> dict:
        uniform = sample['uniform']
        assert torch.all(uniform <= 1.0), "Uniform values must be <= 1, received {}".format(torch.max(uniform))
        assert torch.all(uniform >= 0.0), "Uniform values must be >= 0, received {}".format(torch.max(uniform))
        uniform = torch.clamp(uniform, eps, 1-eps)
        gumbel = -torch.log(-torch.log(uniform))
        sample['uniform'] = gumbel
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample:dict) -> dict:
        for key in sample.keys():
            sample[key] = torch.tensor(sample[key], dtype=torch.float32)
        sample['uniform'] = torch.permute(sample['uniform'], (2, 0, 1))
        return sample
    
class Resize(object):
    """Resize uniform data to image_shape."""
    def __init__(self, image_shape:tuple[int, int]):
        self.image_shape = image_shape

    def __call__(self, sample:dict) -> dict:
        uniform = sample['uniform']
        uniform = transforms.functional.resize(
            uniform, self.image_shape
        )
        sample['uniform'] = uniform
        return sample
    
class Pad(object):
    """Pad uniform data."""
    def __init__(self, padding_mode:str, paddings=(0, 0, 1, 1)):
        self.padding_mode = padding_mode
        self.paddings = paddings

    def __call__(self, sample:dict) -> dict:
        uniform = sample['uniform']
        uniform = transforms.functional.pad(
            uniform, self.paddings, padding_mode=self.padding_mode
        )
        sample['uniform'] = uniform
        return sample


# dataset class
class StormDataset(Dataset):
    def __init__(self, data_dict:dict[str, np.ndarray], transform=None):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.length = len(data_dict[self.keys[0]])
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch = {}
        for key in self.keys:
            batch[key] = self.data[key][idx]
        if self.transform:
            batch = self.transform(batch)
        return batch


def load_data(datadir:str, batch_size:int, padding_mode:str="reflect",
              img_size:tuple=(18, 22)) -> tuple[Dataset, Dataset, dict]:
    traindata, validdata, metadata = prep_xr_data(datadir)
    train = StormDataset(sample_dict(traindata))
    valid = StormDataset(sample_dict(validdata))

    transform = transforms.Compose(
        [ToTensor(), Gumbel(), Resize(img_size),
         Pad(padding_mode, (0, 0, 2, 2))]
         )
    
    train = StormDataset(sample_dict(traindata), transform=transform)
    valid = StormDataset(sample_dict(validdata), transform=transform)

    trainsampler = WeightedRandomSampler(train.data['weight'], len(train), replacement=True)
    trainloader = DataLoader(train, batch_size=batch_size, pin_memory=True, sampler=trainsampler)
    validloader = DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True)

    return trainloader, validloader, metadata


# %% TEST FUNCTIONS BEHAVING ##################################################
def test_sampling_ratios(dataloader):
    """Check sampling ratios of a dataloader."""
    sample = next(iter(dataloader))['label'].numpy()
    labels = Counter(sample)

    for sample in dataloader:
        labels += Counter(sample['label'].numpy())

    return labels

def test_iter_time(dataloader):
    """Time taken to iterate over dataset."""
    start = time.time()
    i = 0
    nims = 0
    for batch in dataloader:
        i += 1
        nims += batch['uniform'].shape[0]
    end = time.time()
    print("Time taken to iterate over dataset ({:,.0f} batches, {:,.0f} samples): {:.2f} seconds.".format(i, nims, end - start))
# %% ##########################################################################