import math
import numpy as np
import tensorflow as tf


def sample(data, size:int, replace=False)->list:
    idx = np.random.choice(range(len(data)), size=size, replace=replace)
    return tf.gather(data, idx, axis=0)

    
class BalancedBatch(tf.keras.utils.Sequence):
    """Data loader for balanced batching of arbitrary data classes."""
    #* There's a way to do this better with .repeat()
    def __init__(self, datasets:list, ratios:list,
                 batch_size=64, infinite=False,
                 labels:list=None, **kwargs):
        super().__init__(**kwargs)
        if infinite:
            print('Creating infinite balanced batch generator')
        assert len(datasets) == len(ratios), f"Number of datasets ({len(datasets)})"\
            f"and number of ratios ({len(ratios)}) must be the same."
        if labels is not None:
            assert len(datasets) == len(labels), f"Number of datasets ({len(datasets)})"\
                f"and number of labels ({len(labels)}) must be the same."
        assert np.isclose(sum(ratios), 1), "ratios must sum to 1."
        self.datasets = datasets
        self.inf = infinite
        self.conditions = labels
        self.batch_sizes = [int(ratio * batch_size) for ratio in ratios]
        self.size = sum([len(dataset) for dataset in datasets])
        self.steps = self.size // batch_size #? unsure

    def number_of_conditions(self):
        return len(self.conditions)

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> tuple:
        """Return one balanced batch of data."""
        if not self.inf:
            if index >= self.__len__():
                raise StopIteration
        
        batches = []

        if self.labels is None:
            for dataset, batch_size in zip(self.datasets, self.batch_sizes):
                batches.append(sample(dataset, batch_size, replace=True))
            balanced_batch = tf.concat(batches, axis=0)
            return balanced_batch
        else:
            conditions = []
            for dataset, condition, batch_size in zip(self.datasets, self.conditions, self.batch_sizes):
                batches.append(sample(dataset, batch_size, replace=True))
                conditions.append([condition] * batch_size)
            balanced_batch = tf.concat(batches, axis=0)
            conditions = tf.concat(conditions, axis=0)
            return balanced_batch, conditions

        