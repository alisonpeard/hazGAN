import time
import numpy as np
from collections import Counter
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from ..data import load_xr_data
from ..data import sample_dict
from ..constants import TEST_YEAR

SUBSET_METHOD = 'equal' # should be pre_only, change to equal on machines with low RAM

__all__ = ['load_data', 'test_sampling_ratios', 'test_iter_time']

# Transforms
class Gumbel(object):
    """Convert uniform data to Gumbel using PIT."""
    def __call__(self, sample:dict, eps:float=1e-6) -> dict:
        uniform = sample['uniform']
        assert torch.all(uniform <= 1.0),"Uniform values must be <= 1, received {}".format(torch.max(uniform))
        assert torch.all(uniform >= 0.0), "Uniform values must be >= 0, received {}".format(torch.max(uniform))
        clamped = torch.clamp(uniform, eps, 1-eps)
        gumbel = -torch.log(-torch.log(clamped))
        sample['uniform'] = gumbel
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample:dict) -> dict:
        sample['uniform'] = torch.tensor(sample['uniform'], dtype=torch.float32)

        # permute to (C, H, W)
        if len(sample['uniform'].shape) == 3:
            sample['uniform'] = torch.permute(sample['uniform'], (2, 0, 1))
        elif len(sample['uniform'].shape) == 4:
            sample['uniform'] = torch.permute(sample['uniform'], (0, 3, 1, 2))

        # reshape condition for dense layer
        sample['condition'] = torch.tensor(sample['condition'], dtype=torch.float32).reshape(-1,)
        sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
        sample['weight'] = torch.tensor(sample['weight'], dtype=torch.float32)
        sample['season'] = torch.tensor(sample['season'], dtype=torch.long)
        sample['days_since_epoch'] = torch.tensor(sample['days_since_epoch'], dtype=torch.long)

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
    def __init__(self, padding_mode:str, paddings=(1, 1, 1, 1)):
        self.padding_mode = padding_mode
        self.paddings = paddings

    def __call__(self, sample:dict) -> dict:
        uniform = sample['uniform']
        uniform = transforms.functional.pad(
            uniform, self.paddings, padding_mode=self.padding_mode
        )
        sample['uniform'] = uniform
        return sample


class sendToDevice(object):
    """Cast uniform data to float32."""
    def __init__(self, device:str) -> None:
        self.device = device
    
    def __call__(self, sample:dict) -> dict:
        for key in sample.keys():
            sample[key] = sample[key].to(self.device)
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
        return self.take(idx)

    def take(self, idx:int):
        """Works for integers and slices."""
        sample = {}
        for key in self.keys:
            sample[key] = self.data[key][idx]
        if self.transform and (not sample.get('transformed', False)):
            sample = self.transform(sample)
        return sample
            
    @staticmethod
    def filterdict(datadict, key:str):
        classes = list(set(datadict[key]))
        classdicts = []
        for label in classes:
            # indices = torch.nonzero(datadict[key] == label)
            indices = np.nonzero(datadict[key] == label)
            classdict = {key: values[indices] for key, values in datadict.items()}
            classdicts.append(classdict)
        return classdicts
        
    @staticmethod
    def subsetdict(datadict, size:int):
        """Return a subset of a dictionary."""
        n = len(datadict[list(datadict.keys())[0]])
        indices = np.random.choice(n, size, replace=True)
        return {key: datadict[key][indices] for key in datadict.keys()}
    
    @staticmethod
    def concatdicts(classdicts:list):
        datadict = {}
        for key in classdicts[0].keys():
            values = []
            for classdict in classdicts:
                values.append(classdict[key])

            if isinstance(values[0], np.ndarray):
                values = np.concatenate(values)

            elif isinstance(values[0], torch.Tensor):
                values = torch.stack(values)
                    
            datadict[key] = values

        return datadict


    def subset(self, size:int, sampling:str=SUBSET_METHOD, verbose:bool=True) -> 'StormDataset':
        """Return a subset of the dataset."""
        if verbose:
            print(f"\nSubsetting dataset to size {size} using {sampling} sampling.\n")
        n = len(self)
        if size > n:
            return self
        
        classdicts = self.filterdict(self.data, 'label')
        nclasses = len(classdicts)

        if sampling == 'pre_only':
            class_size = size
            print(f"\nSampling {class_size} samples from the first class only.\n")
            newdict = self.subsetdict(classdicts[0], class_size)
            
            newdicts = [newdict]
            for i in range(len(classdicts) - 1):
                newdicts.append(classdicts[i + 1])
        else:
            if sampling == 'equal':
                class_size = size // nclasses # even sampling
                class_sizes = [class_size] * nclasses

            elif sampling == 'proportional':
                class_props = [len(classdicts[i]['label']) / n for i in range(nclasses)]
                class_sizes = [int(size * prop) for prop in class_props]
                
            newdicts = []
            for classdict, class_size in zip(classdicts, class_sizes):
                newdict = self.subsetdict(classdict, class_size)
                newdicts.append(newdict)

        if verbose:
            for newdict in newdicts:
                print(f"Label: {newdict['label'][0]}")
                print(f"Class size: {len(newdict['label'])}")
        
        newdict = self.concatdicts(newdicts)
        return StormDataset(newdict, transform=self.transform)
    

    def pretransform(self, transform=None):
        """Pre-transform data"""
        transformed_data = []
        for idx in (pbar := tqdm(range(self.length))):
            pbar.set_description('Pre-transforming data')
            sample = self[idx]
            sample['transformed'] = torch.tensor(True)
            transformed_data.append(sample)
        newdict = self.concatdicts(transformed_data)
        return StormDataset(newdict, transform=transform)
        


def load_data(datadir:str, batch_size:int, padding_mode:str="reflect",
              img_size:tuple=(18, 22), device='mps', train_size:float=0.8,
              fields:list=['u10', 'tp'], epoch='1940-01-01', verbose=True,
              thresholds:list=[15, np.inf], testyear:int=TEST_YEAR,
              cache:bool=True, subset:int=None) -> tuple[Dataset, Dataset, dict]:
    """Load data and return train and valid dataloaders."""
    
    traindata, validdata, metadata = load_xr_data(
        datadir, train_size=train_size, fields=fields, epoch=epoch,
        verbose=verbose, testyear=testyear, thresholds=thresholds,
        cache=cache
        )

    pretransforms = transforms.Compose([
        ToTensor(),
        Resize(img_size),
        Gumbel(),
        Pad(padding_mode, (1, 1, 1, 1)),
        sendToDevice(device)
        ])
    
    train = StormDataset(sample_dict(traindata), transform=pretransforms)
    valid = StormDataset(sample_dict(validdata), transform=pretransforms)

    if subset:
        assert isinstance(subset, int), "subset must be an integer."
        train = train.subset(subset)

    train = train.pretransform()
    valid = valid.pretransform()

    # had to modify this to not make weights double automatically
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