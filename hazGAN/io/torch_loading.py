import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torchvision import transforms
from environs import Env
import torch

if __name__ == "__main__":
    from base import prep_xr_data, sample_dict
else:
    from .base import prep_xr_data, sample_dict


# Transforms
class Gumbel(object):
    """Convert uniform data to Gumbel using PIT."""
    def __call__(self, sample:dict, eps:float=1e-8) -> dict:
        uniform = sample['uniform']
        assert torch.all(uniform < 1.0), "Uniform values must be < 1, received {}".format(torch.max(uniform))
        assert torch.all(uniform > 0.0), "Uniform values must be > 0, received {}".format(torch.max(uniform))
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
class DictDataset(Dataset):
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



# %% DEV // DEBUGGING BELOW HERE ##############################################
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

# %%
if __name__ == "__main__":
    print('Testing io.py...')
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    traindata, validdata, metadata = prep_xr_data(datadir)
    traindata['uniform'].max()

    # %%make datasets
    transform = transforms.Compose([ToTensor(), Gumbel(), Resize((18, 22)), Pad('reflect', (0, 0, 1, 1))])
    train = DictDataset(sample_dict(traindata))
    valid = DictDataset(sample_dict(validdata))



    # %% make loaders
    trainsampler = WeightedRandomSampler(train.data['weight'], len(train), replacement=True)
    trainloader = DataLoader(train, batch_size=16, pin_memory=True, sampler=trainsampler)
    validloader = DataLoader(valid, batch_size=16, shuffle=False, pin_memory=True)

    # %% 
    next(iter(trainloader))['uniform']
    # %%
    # %%transform wrapper
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# %% Transforms dev




# %%


def create_dataloaders(
    sample_dict: Dict[str, np.ndarray],
    labels: List[Any],
    label_ratios: Dict[Any, float],
    batch_size: int,
    image_shape: Tuple[int, int],
    padding_mode: str = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    
    # Create datasets
    dataset = CustomDataset(sample_dict)
    
    # Split into train/valid
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    
    # Split train by labels
    label_datasets = []
    data_sizes = {}

    return train_dataset
    print("\nCalculating input class sizes...")
    for label in labels:
        label_indices = [i for i in range(len(train_dataset)) 
                        if train_dataset[i]['label'] == label]
        label_datasets.append(Subset(train_dataset, label_indices))
        data_sizes[label] = len(label_indices)
    
    print("\nClass sizes:\n------------")
    for label, size in data_sizes.items():
        print(f"Label: {label} | size: {size:,}")
    
    # Create resampled dataset
    target_dist = list(label_ratios.values())
    train_dataset = ResampledDataset(label_datasets, target_dist)
    
    # Create transform
    transform = TransformWrapper(
        image_shape=image_shape,
        use_gumbel=True,
        padding_mode=padding_mode
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    gc.enable()
    gc.collect()
    
    return train_loader, valid_loader



