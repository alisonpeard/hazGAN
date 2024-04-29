"""
TODO: final holder class to do predictions and PIT tranforms...

Methods:
--------
* identify events in data and extract maxima (all the stuff in R)
* transform data to uniform space
* train
* load weights
* predict
* inverse transform
"""
from .WGAN import WGAN, compile_wgan
from .DCGAN import DCGAN
from .extreme_value_theory import POT

class hazGAN(object):
    """Not finished, long term plan..."""
    def __init__(self, config, datadir, nchannels=2):
        if config.model == 'WGAN':
            self.model = compile_wgan(config, nchannels)
        elif config.model == 'DCGAN':
            pass
        else:
            raise ValueError(f"Invalid model '{config.model}' in config.")
        x, u, m, z, params = self.load_training(datadir, config.train_size, zero_pad=False)
        self.x = x
        self.u = u
        self.medians = m
        self.z = z
        self.params = params

    def __call__(self, n):
        u = self.model(n)
        u_train = self.u[0]
        x_train = self.x[0]
        x = POT.inv_probability_integral_transform(u,u_train, x_train, self.params)
        x += self.medians
        return x
