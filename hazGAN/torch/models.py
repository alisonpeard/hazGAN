"""
Conditional Wasserstein GAN with gradient penalty (cWGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
# %%
from functools import partial
from torch import nn, cat

from .blocks import ResidualUpBlock
from .blocks import ResidualDownBlock
from .blocks import GumbelBlock

__all__ = ['Generator', 'Critic']

    
def _combine(x, label, condition, policy='add'):
    if policy == 'add':
        x += label + condition
    elif policy == 'concat':
        x = cat([x, label, condition], dim=1)
    else:
        raise ValueError(f"Unknown policy {policy}, try 'add', or 'concat'.")
    return x


class Generator(nn.Module):
    def __init__(self, nfields, channel_multiplier, width,
    input_policy, relu, embedding_depth, nconditions,
    latent_dims,
    dropout=None, noise_sd=None, bias=False,
    **kwargs) -> None:
        super(Generator, self).__init__()

        K = channel_multiplier

        # set up feature widths
        self.nfields = nfields

        assert width % 8 == 0, "generator width must be divisible by 8"
        assert width >= 64, "generator width must be at least 64"

        self.width0 = width
        self.width1 = width // 2
        self.width2 = width // 3
        self.latent_dim = latent_dims

        # input handling
        self.input_factor = 1 if input_policy == 'add' else 3
        self.combine_inputs = partial(_combine, policy=input_policy)

        self.constant_to_features = None # placeholder for later

        self.label_to_features = nn.Sequential(
            # input shape: (batch_size,)
            nn.Embedding(nconditions, embedding_depth, sparse=False),
            nn.Linear(embedding_depth, self.width0 * 5 * 5 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(relu),
            nn.BatchNorm2d(self.width0 * nfields),
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.condition_to_features = nn.Sequential(
            # input shape: (batch_size, 1), linear expects 2d input
            nn.Linear(1, embedding_depth, bias=bias),
            nn.Linear(embedding_depth, self.width0 * 5 * 5 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(relu),
            nn.BatchNorm2d(self.width0 * nfields)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.latent_to_features = nn.Sequential(
            # input shape: (batch_size, latent_dim)
            nn.Linear(self.latent_dim, self.width0 * 5 * 5 * nfields, bias=bias), # custom option
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(relu),
            nn.BatchNorm2d(self.width0 *  nfields)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        # f(x) = (x-1)*s + k
        reskws = dict(bias=bias, dropout=dropout, noise_sd=noise_sd)
        self.features_to_image = nn.Sequential(
            ResidualUpBlock(self.input_factor * self.width0 * nfields, self.width1, (3, 3), **reskws),
            # (3, 3; 1) kernel: 5 x 5 -> 7 x 7 -- USING
            # (2, 2; 2) kernel: 5 x 5 -> 10 x 10
            ResidualUpBlock(self.width1, self.width1, (2, 4), 2, **reskws),
            # (3, 4) kernel: 7 x 7 -> 9 x 10
            # (2, 4; 2) kernel: 7 x 7 -> 14 x 16 -- USING
            # (2, 2; 2) kernel: 10 x 10 -> 20 x 20
            # (4, 6; 1) kernel: 10 x 10 -> 13 x 15
            ResidualUpBlock(self.width1, self.width2, (3, 4), **reskws),
            ResidualUpBlock(self.width2, self.width2, (3, 4), **reskws),
            # (2, 2; 2) kernel: 9 x 10 -> 18 x 22
            # (4, 6; 1) kernel: 9 x 10 -> 12 x 15
            # (4, 6; 2) kernel: 9 x 10 -> 20 x 24
            # (3, 4; 1) kernel: 14 x 16 -> 16 x 19 -- USING
            # (3, 4; 1) kernel: 16 x 19 -> 18 x 22 -- USING
            ResidualUpBlock(self.width2, nfields, (3, 3), **reskws)
            # nn.ReflectionPad2d((1, 1, 1, 1))
            # padding 18 x 22 -> 20 x 24
        ) # output shape: (batch_size, 20, 24, nfields)

        self.refine_fields = nn.Sequential(
            nn.Conv2d(nfields, K * nfields, kernel_size=4, padding="same", groups=nfields),
            nn.LeakyReLU(relu),
            nn.BatchNorm2d(K * nfields),
            nn.Conv2d(K * nfields, nfields, kernel_size=3, padding="same"),
            GumbelBlock(nfields)
        ) # output shape: (batch_size, 20, 24, nfields)


    def forward(self, z, label, condition):
        z = self.latent_to_features(z)
        label = self.label_to_features(label)
        condition = self.condition_to_features(condition)
        x = self.combine_inputs(z, label, condition)
        x = self.features_to_image(x)
        x = self.refine_fields(x)
        return x


class Critic(nn.Module):
    def __init__(self, nfields, channel_multiplier, width,
    input_policy, relu, embedding_depth, nconditions,
    latent_dims,
    dropout=None, noise_sd=None, bias=False,
    **kwargs) -> None:
        super(Critic, self).__init__()

        K = channel_multiplier

        # set up feature widths
        self.nfields = nfields
        assert width % 8 == 0, "critic width must be divisible by 8"
        assert width >= 64, "critic width must be at least 64"
        
        self.width0 = width // 3
        self.width1 = width // 2
        self.width2 = width

        # input handling
        self.input_factor = 1 if input_policy == 'add' else 3
        self.combine_inputs = partial(_combine, policy=input_policy)

        self.process_fields = nn.Sequential(
            nn.Conv2d(nfields, K * self.width0 * nfields, kernel_size=4, padding="same", groups=nfields),
            nn.LeakyReLU(relu),
            nn.LayerNorm((K * self.width0 * nfields, 20, 24)),
            nn.Conv2d(K * self.width0 * nfields, self.width0 * nfields, kernel_size=3, padding="same"),
            nn.LeakyReLU(relu),
            nn.LayerNorm((self.width0 * nfields, 20, 24))
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.label_to_features = nn.Sequential(
            nn.Embedding(nconditions, embedding_depth, sparse=False),
            nn.Linear(embedding_depth, self.width0 * 20 * 24 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(relu),
            nn.LayerNorm((self.width0 * nfields, 20, 24))
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.condition_to_features = nn.Sequential(
            nn.Linear(1, embedding_depth, bias=bias),
            nn.Linear(embedding_depth, self.width0 * 20 * 24 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(relu),
            nn.LayerNorm((self.width0 * nfields, 20, 24))
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        # f(x) = (x-k)/s + 1
        reskws = dict(bias=bias, dropout=dropout, noise_sd=noise_sd)
        self.image_to_features = nn.Sequential(
            # ResidualDownBlock(self.input_factor * self.width0 * nfields, self.width1, (4, 5), 2, bias=False),
            # ResidualDownBlock(self.width1, self.width2, (3, 4), bias=False),
            # ResidualDownBlock(self.width2, self.width2, (3, 3), bias=False),
            ResidualDownBlock(self.input_factor * self.width0 * nfields, self.width0, (3, 3), **reskws),
            ResidualDownBlock(self.width0, self.width1, (3, 4), **reskws),
            ResidualDownBlock(self.width1, self.width1, (3, 4), **reskws),
            ResidualDownBlock(self.width1, self.width2, (2, 4), 2, **reskws),
            ResidualDownBlock(self.width2, self.width2, (3, 3), **reskws),
        ) # output shape: (batch_size, width2, 5, 5)


        self.features_to_score = nn.Sequential(
            nn.Conv2d(self.width2, nfields, kernel_size=4, groups=nfields, bias=bias, padding='same'),
            nn.Flatten(1, -1),
            nn.LeakyReLU(relu),
            nn.LayerNorm((nfields * 5 * 5)),
            nn.Linear(nfields * 5 * 5, K * nfields),
            nn.LeakyReLU(relu),
            nn.LayerNorm((K * nfields)),
            nn.Linear(K * nfields, 1),
            nn.Sigmoid() # maybe, makes generator and critic very different scales
        ) # output shape: (batch_size, 1)

    def forward(self, x, label, condition):
        x = self.process_fields(x)
        label = self.label_to_features(label)
        condition = self.condition_to_features(condition)
        x = self.combine_inputs(x, label, condition)
        x = self.image_to_features(x)
        x = self.features_to_score(x)
        return x
    
#%%
