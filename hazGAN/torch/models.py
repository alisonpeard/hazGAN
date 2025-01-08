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


def initialize_weights(self):
    raise NotImplementedError("This function is not used in this module.")
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, nfields, channel_multiplier, width,
    input_policy, lrelu, embedding_depth, nconditions,
    latent_dims, momentum,
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
            nn.BatchNorm1d(embedding_depth, momentum=momentum),
            nn.LeakyReLU(lrelu),
            nn.Linear(embedding_depth, self.width0 * 5 * 5 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.BatchNorm2d(self.width0 * nfields, momentum=momentum),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.condition_to_features = nn.Sequential(
            # input shape: (batch_size, 1), linear expects 2d input
            nn.Linear(1, embedding_depth, bias=bias),
            nn.BatchNorm1d(embedding_depth, momentum=momentum),
            nn.LeakyReLU(lrelu),
            nn.Linear(embedding_depth, self.width0 * 5 * 5 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.BatchNorm2d(self.width0 * nfields, momentum=momentum),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.latent_to_features = nn.Sequential(
            # input shape: (batch_size, latent_dim)
            nn.Linear(self.latent_dim, embedding_depth * nfields, bias=bias),
            nn.BatchNorm1d(embedding_depth * nfields, momentum=momentum),
            nn.LeakyReLU(lrelu),
            nn.Linear(embedding_depth * nfields, 5 * 5 * self.width0 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.BatchNorm2d(self.width0 * nfields, momentum=momentum),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        # f(x) = (x-1)*s + k
        reskws = dict(bias=bias, dropout=dropout, noise_sd=noise_sd, momentum=momentum, lrelu=lrelu)
        self.features_to_image = nn.Sequential(
            ResidualUpBlock(self.input_factor * self.width0 * nfields, self.width1, (3, 3), **reskws),
            ResidualUpBlock(self.width1, self.width1, (2, 4), 2, **reskws),
            ResidualUpBlock(self.width1, self.width2, (3, 4), **reskws),
            ResidualUpBlock(self.width2, self.width2, (3, 4), **reskws),
            ResidualUpBlock(self.width2, nfields, (3, 3), **reskws)
        ) # output shape: (batch_size, 20, 24, nfields)

        self.refine_fields = nn.Sequential(
            nn.Conv2d(nfields, K * nfields, kernel_size=4, padding="same", groups=nfields),
            nn.BatchNorm2d(K * nfields, momentum=momentum),
            nn.LeakyReLU(lrelu),
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
    input_policy, embedding_depth, nconditions,
    lrelu,
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
            nn.LayerNorm((K * self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(lrelu),
            nn.Conv2d(K * self.width0 * nfields, self.width0 * nfields, kernel_size=3, padding="same"),
            nn.LayerNorm((self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.label_to_features = nn.Sequential(
            nn.Embedding(nconditions, embedding_depth, sparse=False),
            nn.LayerNorm(embedding_depth),
            nn.LeakyReLU(lrelu),
            nn.Linear(embedding_depth, self.width0 * 20 * 24 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LayerNorm((self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.condition_to_features = nn.Sequential(
            nn.Linear(1, embedding_depth, bias=bias),
            nn.LayerNorm(embedding_depth),
            nn.LeakyReLU(lrelu),
            nn.Linear(embedding_depth, self.width0 * 20 * 24 * nfields, bias=bias),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LayerNorm((self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(lrelu)
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        # f(x) = (x-k)/s + 1)
        reskws = dict(bias=bias, dropout=dropout, noise_sd=noise_sd, lrelu=lrelu)
        self.image_to_features = nn.Sequential(
            ResidualDownBlock(self.input_factor * self.width0 * nfields, self.width0, (3, 3), **reskws),
            ResidualDownBlock(self.width0, self.width1, (3, 4), **reskws),
            ResidualDownBlock(self.width1, self.width1, (3, 4), **reskws),
            ResidualDownBlock(self.width1, self.width2, (2, 4), 2, **reskws),
            ResidualDownBlock(self.width2, self.width2, (3, 3), **reskws),
        ) # output shape: (batch_size, width2, 5, 5)


        self.features_to_score = nn.Sequential(
            nn.Conv2d(self.width2, nfields, kernel_size=4, groups=nfields, bias=bias, padding='same'),
            nn.Flatten(1, -1),
            nn.LayerNorm((nfields * 5 * 5)),
            nn.LeakyReLU(lrelu),
            nn.Linear(nfields * 5 * 5, K * nfields),
            nn.LayerNorm((K * nfields)),
            nn.LeakyReLU(lrelu),
            nn.Linear(K * nfields, 1)
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
