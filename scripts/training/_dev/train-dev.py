import os
import wandb
import tensorflow as tf
import hazGAN as hg

import numpy as np
import tensorflow as tf

def sample(data, size:int, replace=False)->list:
    idx = np.random.choice(range(len(data)), size=size, replace=replace)
    return [data[i, ...] for i in idx]


class BalancedBatch(tf.keras.utils.Sequence):
    """Data loader that returns balanced batches."""
    def __init__(self, majority, minority, batch_size):
        self.majority = majority
        self.minority = minority
        self.batch_size = batch_size
        self.n = len(majority) + len(minority)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index)->tuple:
        """Return one batch of data."""
        n_each = self.batch_size // 2
        maj_batch = sample(self.majority, size=n_each)
        min_batch = sample(self.minority, size=n_each, replace=True)
        return tf.concat(maj_batch + min_batch, axis=0),


wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync') 
datadir = os.path.join(wd, 'training', "18x22")  # keep data folder in parent directory
    
wandb.init(project="test", mode="disabled")
wandb.config.update({
    'nepochs': 1,
    'train_size': 128,
    'batch_size': 128,
    'chi_frequency': 1
    },
    allow_val_change=True)
runname = 'dry-run'

config = wandb.config

minority = hg.load_training(datadir,
                        config.train_size,
                        padding_mode='reflect',
                        gumbel_marginals=config.gumbel,
                        u10_min=15
                        )
train_minority = minority['train_u']
test_minority = minority['test_u']

majority = hg.load_training(datadir,
                        config.train_size,
                        padding_mode='reflect',
                        gumbel_marginals=config.gumbel,
                        u10_max=15
                        )
train_majority = majority['train_u']
test_majority = majority['test_u']


train = BalancedBatch(train_majority, train_minority, 64)
test = BalancedBatch(test_majority, test_minority, 64)

x = tf.data.Dataset.from_tensor_slices(train_majority).batch(config.batch_size)