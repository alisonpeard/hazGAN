import numpy as np
import tensorflow as tf

def sample(data, size:int, replace=False)->list:
    idx = np.random.choice(range(len(data)), size=size, replace=replace)
    return tf.gather(data, idx, axis=0)


class BalancedBatch(tf.keras.utils.Sequence):
    """Data loader that returns balanced batches."""
    def __init__(self, majority, minority, batch_size, ratio=0.5, name='training'):
        self.name = name
        self.majority = majority
        self.minority = minority
        self.batch_size = batch_size
        self.min_batch_size = int(ratio * batch_size)
        self.maj_batch_size = int((1 - ratio) * batch_size)
        self.n = len(majority) + len(minority)
        self.steps = self.n // self.batch_size

    def __len__(self):
        return self.steps

    def __getitem__(self, index) -> tuple:
        """Return one batch of data."""
        if index >= self.__len__():
            raise StopIteration
        min_batch = sample(self.minority, size=self.min_batch_size, replace=True)
        maj_batch = sample(self.majority, size=self.maj_batch_size)
        batch = tf.concat([maj_batch, min_batch], axis=0)
        return batch,

    
