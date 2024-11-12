import math
import numpy as np
import tensorflow as tf

def sample(data, size:int, replace=False)->list:
    idx = np.random.choice(range(len(data)), size=size, replace=replace)
    return tf.gather(data, idx, axis=0)

    
class BalancedBatchNd(tf.keras.utils.Sequence):
    """Data loader for balanced batching of arbitrary data classes."""
    def __init__(self, datasets:list, ratios:list,
                 batch_size=64, infinite=False, **kwargs):
        super().__init__(**kwargs)
        if infinite:
            print('Creating infinite balanced batch generator')
        assert len(datasets) == len(ratios), f"Number of datasets ({len(datasets)})"\
            f"and number of ratios ({len(ratios)}) must be the same."
        assert np.isclose(sum(ratios), 1), "ratios must sum to 1."
        self.datasets = datasets
        self.inf = infinite
        self.batch_sizes = [int(ratio * batch_size) for ratio in ratios]
        self.size = sum([len(dataset) for dataset in datasets])
        self.steps = self.size // batch_size # unsure

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> tuple:
        """Return one balanced batch of data."""
        if not self.inf:
            if index >= self.__len__():
                raise StopIteration
        
        batches = []
        for dataset, batch_size in zip(self.datasets, self.batch_sizes):
            batches.append(sample(dataset, batch_size, replace=True))
        balanced_batch = tf.concat(batches, axis=0)
        return balanced_batch,
        

class BalancedBatch2d(tf.keras.utils.Sequence):
    """Data loader that returns balanced batches."""
    def __init__(self, majority, minority, batch_size, ratio=0.5,
                 name='training', infinite=False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.majority = majority
        self.minority = minority
        self.batch_size = batch_size
        self.min_batch_size = int(ratio * batch_size)
        self.maj_batch_size = int((1 - ratio) * batch_size)
        self.n = len(majority) + len(minority) if not infinite else np.inf
        self.steps = self.n // self.batch_size 

    def __len__(self):
        return self.steps

    def __getitem__(self, index) -> tuple:
        """Return one balanced batch of data."""
        if index >= self.__len__():
            raise StopIteration
        
        min_batch = sample(self.minority, size=self.min_batch_size, replace=True)
        maj_batch = sample(self.majority, size=self.maj_batch_size)
        balanced_batch = tf.concat([maj_batch, min_batch], axis=0)
        return balanced_batch,