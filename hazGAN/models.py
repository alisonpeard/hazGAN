"""
Conditional Wasserstein GAN with gradient penalty (cWGAN-GP).

References:
..[1] Gulrajani (2017) https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py 
..[2] Harris (2022) - application
"""
# %%
import warnings
import functools
import torch
from torch import nn
from inspect import signature
from typing import Tuple, Union
from pytorch.blocks import (
    ResidualUpBlock,
    ResidualDownBlock,
    GumbelBlock
)

# from .statistics import chi_loss, inv_gumbel
# from .pytorch import DiffAugment
# from .pytorch import wrappers

SAMPLE_CONFIG = {
    'generator_width': 64,
    'nconditions': 2,
    'embedding_depth': 64,
    'latent_dims': 64,
    'lrelu': 0.2,
    'critic_width': 64
}

# %%
def sample_gumbel(shape, eps=1e-20, temperature=1., offset=0., seed=None, device='mps'):
    """Sample from Gumbel(0, 1)"""
    T = torch.tensor(temperature, device=device)
    O = torch.tensor(offset, device=device)
    U = torch.rand(shape, device=device)
    return O - T * torch.log(-torch.log(U + eps) + eps)


def initialise_variables():
    """Allows me to specify distribution in wandb sweeps."""
    torch.gumbel = sample_gumbel
    torch.uniform = torch.rand
    torch.normal = torch.randn
    

class Generator(nn.Module):
    def __init__(self, config, nfields=2):
        super(Generator, self).__init__()

        # set up feature widths
        self.nfields = nfields
        width = config['generator_width']
        assert width % 8 == 0, "generator width must be divisible by 8"
        assert width >= 64, "generator width must be at least 64"
        self.width0 = width
        self.width1 = width // 2
        self.width2 = width // 3
        self.latent_dims = config['latent_dims']

        self.constant_to_features = None # placeholder for later

        self.label_to_features = nn.Sequential(
            nn.Embedding(config['nconditions'], config['embedding_depth'], sparse=False),
            nn.Linear(config['embedding_depth'], self.width0 * 5 * 5 * nfields, bias=False),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(config['lrelu']),
            nn.BatchNorm2d(self.width0 * nfields),
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.condition_to_features = nn.Sequential(
            nn.Linear(1, config['embedding_depth'], bias=False),
            nn.Linear(config['embedding_depth'], self.width0 * 5 * 5 * nfields, bias=False),
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(config['lrelu']),
            nn.BatchNorm2d(self.width0 * nfields)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dims, self.width0 * 5 * 5 * nfields, bias=False), # custom option
            nn.Unflatten(-1, (self.width0 * nfields, 5, 5)),
            nn.LeakyReLU(config['lrelu']),
            nn.BatchNorm2d(self.width0 *  nfields)
        ) # output shape: (batch_size, width0 * nfields, 5, 5)

        self.features_to_image = nn.Sequential(
            ResidualUpBlock(self.width0 * nfields, self.width1, (3, 3), bias=False),
            # (3,3) kernel: 5x5 -> 15x15
            ResidualUpBlock(self.width1, self.width2, (3, 4), bias=False),
        
            ResidualUpBlock(self.width2, nfields, (4, 6), 2, bias=False),
        ) # output shape: (batch_size, 20, 24, nfields)

        self.refine_fields = nn.Sequential(
            nn.Conv2d(nfields, nfields, kernel_size=4, padding="same", groups=nfields),
            nn.LeakyReLU(config['lrelu']),
            nn.BatchNorm2d(nfields),
            nn.Conv2d(nfields, nfields, kernel_size=3, padding="same", groups=nfields),
            GumbelBlock(nfields)
        ) # output shape: (batch_size, 20, 24, nfields)


    def forward(self, z, label, condition):
        z = self.latent_to_features(z)
        label = self.label_to_features(label)
        condition = self.condition_to_features(condition)
        x = z + label + condition
        # x = torch.cat([z, label, condition], dim=1)
        x = self.features_to_image(x)
        x = self.refine_fields(x)
        return x

    
class Critic(nn.Module):
    def __init__(self, config, nfields=2):
        super(Critic, self).__init__()

        # set up feature widths
        self.nfields = nfields
        width = config['critic_width']
        assert width % 8 == 0, "critic width must be divisible by 8"
        assert width >= 64, "critic width must be at least 64"
        
        self.width0 = width // 3
        self.width1 = width // 2
        self.width2 = width

        self.process_fields = nn.Sequential(
            nn.Conv2d(nfields, self.width0 * nfields, kernel_size=4, padding="same", groups=nfields),
            nn.LeakyReLU(config['lrelu']),
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.label_to_features = nn.Sequential(
            nn.Embedding(config['nconditions'], config['embedding_depth'], sparse=False),
            nn.Linear(config['embedding_depth'], self.width0 * 20 * 24 * nfields, bias=False),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(config['lrelu']),
            nn.LayerNorm((self.width0 * nfields, 20, 24))
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.condition_to_features = nn.Sequential(
            nn.Linear(1, config['embedding_depth'], bias=False),
            nn.Linear(config['embedding_depth'], self.width0 * 20 * 24 * nfields, bias=False),
            nn.Unflatten(-1, (self.width0 * nfields, 20, 24)),
            nn.LeakyReLU(config['lrelu']),
            nn.LayerNorm((self.width0 * nfields, 20, 24))
        ) # output shape: (batch_size, width0 * nfields, 20, 24)

        self.image_to_features = nn.Sequential(
            ResidualDownBlock(self.width0 * nfields, self.width1, (4, 5), 2, bias=False),
            ResidualDownBlock(self.width1, self.width2, (3, 4), bias=False),
            ResidualDownBlock(self.width2, self.width2, (3, 3), bias=False),
        ) # output shape: (batch_size, width2, 1, 1)


        self.features_to_score = nn.Sequential(
            nn.Conv2d(self.width2, nfields, kernel_size=4, groups=nfields, bias=False, padding='same'),
            nn.Flatten(1, -1),
            nn.LeakyReLU(config['lrelu']),
            nn.LayerNorm((nfields * 5 * 5)),
            nn.Linear(nfields * 5 * 5, 1),
            nn.Sigmoid() # maybe
        ) # output shape: (batch_size, 1)

    def forward(self, x, label, condition):
        x = self.process_fields(x)
        label = self.label_to_features(label)
        condition = self.condition_to_features(condition)
        x = x + label + condition
        x = self.image_to_features(x)
        x = self.features_to_score(x)
        return x
    

if __name__ == "__main__": # tests
    generator = Generator(SAMPLE_CONFIG)
    critic =  Critic(SAMPLE_CONFIG)

    z = torch.rand(1, SAMPLE_CONFIG['latent_dims'])
    label = torch.randint(0, SAMPLE_CONFIG['nconditions'], (1,))
    condition = torch.rand(1, 1)
    x = torch.rand(1, 2, 20, 24)

    x = generator.forward(z, label, condition)
    p = critic.forward(x, label, condition)
    print("Generated shape:", x.shape)
    print("Critic score shape:", p.shape)
#%%

"""
def get_optimizer_kwargs(optimizer):
    optimizer = getattr(optimizers, optimizer)
    params = signature(optimizer).parameters
    return params


def process_optimizer_kwargs(config):
    kwargs = {
        "learning_rate": config['learning_rate'],
        "beta_1": config['beta_1'],
        "beta_2": config['beta_2'],
        "weight_decay": config['weight_decay'],
        "use_ema": config['use_ema'],
        "ema_momentum": config['ema_momentum'],
        "ema_overwrite_frequency": config['ema_overwrite_frequency'],
    }
    params = get_optimizer_kwargs(config['optimizer'])
    kwargs = {key: val for key, val in kwargs.items() if key in params}
    
    return kwargs

def compile_wgan(config, nchannels=2):
    initialise_variables()
    kwargs = process_optimizer_kwargs(config)
    optimizer = getattr(optimizers, config['optimizer'])
    critic_optimizer = optimizer(**kwargs)
    generator_optimizer = optimizer(**kwargs)
    wgan = WGANGP(config, nchannels=nchannels)
    wgan.compile(
        critic_optimizer=critic_optimizer,
        generator_optimizer=generator_optimizer
        )
    return wgan


def printv(message, verbose):
    if verbose:
        tf.print(message)

"""

"""


class WGANGP(keras.Model):
    """Wasserstein GAN with gradient penalty."""

    # this should improve memory usage
    __slots__ = ['critic', 'generator', 'latent_dim', 'lambda_gp',
                    'config', 'latent_space_distn', 'trainable_vars', 'inv', 'augment',
                    'seed', 'chi_rmse_tracker', 'generator_loss_tracker', 'critic_loss_tracker',
                    'value_function_tracker', 'critic_real_tracker', 'critic_fake_tracker',
                    'critic_valid_tracker', 'images_seen', 'critic_steps', 'generator_steps',
                    'critic_grad_norm', 'generator_grad_norm', 'critic_grad_norms',
                    'generator_grad_norms']

    def __init__(self, config, nchannels=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.critic = define_critic(config, nchannels)
        self.generator = define_generator(config, nchannels)
        self.latent_dim = config['latent_dims']
        self.lambda_gp = config['lambda_gp']
        self.config = config
        self.latent_space_distn = getattr(tf.random, config['latent_space_distn'])
        self.trainable_vars = [
            *self.generator.trainable_variables,
            *self.critic.trainable_variables,
        ]
        if config['gumbel']:
            self.inv = inv_gumbel
        else:
            self.inv = lambda x: x # will make serialising etc. difficult
        self.augment = functools.partial(DiffAugment, policy=config['augment_policy'])
        self.seed = config['seed']

        # stateful metrics
        self.chi_rmse_tracker = keras.metrics.Mean(name="chi_rmse")
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.critic_loss_tracker = keras.metrics.Mean(name="critic_loss")
        self.value_function_tracker = keras.metrics.Mean(name="value_function")
        self.critic_real_tracker = keras.metrics.Mean(name="critic_real")
        self.critic_fake_tracker = keras.metrics.Mean(name="critic_fake")
        self.critic_valid_tracker = keras.metrics.Mean(name="critic_valid")
        self.gradient_penalty_tracker = keras.metrics.Mean(name="gradient_penalty")

        # training statistics # ? setting dtype=tf.int32 fails ?
        self.images_seen = keras.metrics.Sum(name="images_seen")
        self.critic_steps = keras.metrics.Sum(name="critic_steps")
        self.generator_steps = keras.metrics.Sum(name="generator_steps")
        self.critic_grad_norm = keras.metrics.Mean(name="critic_grad_norm")
        self.generator_grad_norm = keras.metrics.Mean(name="generator_grad_norm")

        # monitor for vanishing gradients
        self.critic_grad_norms = [
            keras.metrics.Mean(name=f"critic_{i}_{var.path}") for i, var in enumerate(self.critic.trainable_variables)
        ]
        self.generator_grad_norms = [
            keras.metrics.Mean(name=f"generator_{i}_{var.path}") for i, var in enumerate(self.generator.trainable_variables)
        ]


    def compile(self, critic_optimizer, generator_optimizer, *args, **kwargs) -> None:
        super().compile(*args, **kwargs)
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer.build(self.critic.trainable_variables)
        self.generator_optimizer.build(self.generator.trainable_variables)


    def call(self, condition, label, nsamples=5,
             noise=None, temp=1., offset=0, seed=None) -> tf.Tensor:
        """Return uniformly distributed samples from the generator."""
        if noise is None:
            noise = self.latent_space_distn(
                (nsamples, self.latent_dim),
                temperature=temp,
                offset=offset,
                seed=seed
                )
        else:
            n = noise.shape[0]
            assert n == nsamples, f"Latent vector must be same length ({n}) as requested number of samples ({nsamples})."

        raw = self.generator([noise, condition, label], training=False)
        tf.print("Minimum before transformation:", tf.reduce_min(raw))
        tf.print("Maximum before transformation:", tf.reduce_max(raw))
        return self.inv(raw)
    

    def evaluate(self, x, **kwargs) -> dict:
        """Overwrite evaluation function for custom data.
        """
        score_valid = 0
        with warnings.catch_warnings(): # suppress out of range error
            warnings.filterwarnings("ignore", message="Local rendezvous")
            for n, batch in enumerate(x):
                try:
                    data = batch['uniform']
                    condition = batch["condition"]
                    label = batch["label"]
                    critic_score = self.critic([data, condition, label], training=False)
                    score_valid += tf.reduce_mean(critic_score)
                except tf.errors.OutOfRangeError:
                    break
        score_valid = score_valid / (n + 1)
        self.critic_valid_tracker.update_state(score_valid)
        return {'critic': self.critic_valid_tracker.result()}


    def train_critic(self, data, condition, label, batch_size) -> None:
        """Train critic with gradient penalty.
        
        Debugging:
            >> concrete_fn = self.train_step.get_concrete_function(dict(
                uniform=tf.TensorSpec(shape=(None,20,24,2)),
                condition=tf.TensorSpec(shape=(None,)),
                label=tf.TensorSpec(shape=(None,))
                ))
            >> print(self.train_step.pretty_printed_concrete_signatures())
            >> graph = concrete_fn.graph
            >> for node in graph.as_graph_def().node:
                    print(f'{node.input} -> {node.name}')
        """
        print("\nTracing critic...")
        random_noise = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator([random_noise, condition, label], training=False)

        with tf.GradientTape() as tape:
            score_real = self.critic([self.augment(data), condition, label])
            score_fake = self.critic([self.augment(fake_data), condition, label])
            critic_loss = tf.reduce_mean(score_fake) - tf.reduce_mean(score_real) # value function (observed to correlate with sample quality --Gulrajani 2017)
            eps = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
            differences = fake_data - data
            interpolates = data + (eps * differences)  # interpolated data

            with tf.GradientTape() as tape_gp:
                tape_gp.watch(interpolates)
                score = self.critic([interpolates, condition, label])
            gradients = tape_gp.gradient(score, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            print("Shape of interpolated gradients:", slopes.shape)

            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            critic_loss += self.lambda_gp * gradient_penalty

        gradients = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_weights))

        self.critic_loss_tracker(critic_loss)
        self.value_function_tracker.update_state(-critic_loss)
        self.critic_real_tracker.update_state(tf.reduce_mean(score_real))
        self.critic_fake_tracker.update_state(tf.reduce_mean(score_fake))
        self.gradient_penalty_tracker.update_state(gradient_penalty)

        self.critic_steps.update_state(tf.constant(1, dtype=tf.int32))

        # get gradient norms
        global_gradient_norm = tf.linalg.global_norm(gradients)
        self.critic_grad_norm.update_state(global_gradient_norm)

        for i, grad in enumerate(gradients):
            self.critic_grad_norms[i].update_state(tf.linalg.global_norm([grad]))

        return None

    
    def train_generator(self, data, condition, label, batch_size) -> None:
        """https://www.tensorflow.org/guide/function#conditionals
        """
        print("\nTracing generator...\n")
        random_noise = self.latent_space_distn((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_data = self.generator([random_noise, condition, label])
            score = self.critic([self.augment(generated_data), condition, label], training=False)
            generator_loss = -tf.reduce_mean(score)
            generator_penalised_loss = generator_loss
        
        chi_rmse = chi_loss(self.inv(data), self.inv(generated_data))
        gradients = tape.gradient(generator_penalised_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))
        
        self.generator_loss_tracker.update_state(generator_loss)
        self.chi_rmse_tracker.update_state(chi_rmse)

        self.generator_steps.update_state(tf.constant(1, dtype=tf.int32))

        # get gradient norms
        global_gradient_norm = tf.linalg.global_norm(gradients)
        self.generator_grad_norm.update_state(global_gradient_norm)
        
        for i, grad in enumerate(gradients):
            self.generator_grad_norms[i].update_state(tf.linalg.global_norm([grad]))

        return None


    def skip(self, *args, **kwargs) -> None:
        return None
    
   
    @tf.function
    def train_step(self, batch) -> dict:
        """print(train_step.pretty_printed_concrete_signatures())"""
        data = batch['uniform']
        condition = batch['condition']
        label = batch['label']
        batch_size = tf.shape(data)[0] # dynamic for graph mode
        
        # train critic
        self.train_critic(data, condition, label, batch_size)

        metrics = {
            "critic_loss": self.critic_loss_tracker.result(),
            "value_function": -self.value_function_tracker.result(),
            'critic_real': self.critic_real_tracker.result(),
            'critic_fake': self.critic_fake_tracker.result(),
            'gradient_penalty': self.gradient_penalty_tracker.result()
        }

        # train generator
        generator_flag = tf.math.logical_or(tf.math.equal(
            self.critic_steps.result(), 1),
            tf.math.equal(self.critic_steps.result() % self.config['training_balance'], 0)
            )
        train_generator = lambda: self.train_generator(data, condition, label, batch_size)
        skip_generator = lambda: self.skip()
        tf.cond(
            generator_flag,
            train_generator,
            skip_generator
            )
        
        # update metrics
        self.images_seen.update_state(batch_size)
        metrics["generator_loss"] = self.generator_loss_tracker.result()
        metrics['chi_rmse'] = self.chi_rmse_tracker.result()
        metrics['critic_steps'] = self.critic_steps.result()
        metrics['generator_steps'] = self.generator_steps.result()

        metrics["images_seen"] = self.images_seen.result()
        metrics["critic_grad_norm"] = self.critic_grad_norm.result()
        metrics["generator_grad_norm"] = self.generator_grad_norm.result()
        
        i = 0
        for var, metric in zip(self.critic.trainable_variables, self.critic_grad_norms):
            metrics[f"critic_{i}_{var.path}"] = metric.result()
            i += 1

        i = 0
        for var, metric in zip(self.generator.trainable_variables, self.generator_grad_norms):
            metrics[f"generator_{i}_{var.path}"] = metric.result()
            i += 1

        # print logs if in eager mode
        if tf.executing_eagerly():
            print(f"\nBatch mean:", tf.math.reduce_mean(data))
            print(f"Batch std:", tf.math.reduce_std(data))
    
        return metrics

    @property
    def metrics(self) -> list:
        """Define which stateful metrics to reset per-epoch."""
        return [
            self.critic_real_tracker,
            self.critic_fake_tracker,
            self.critic_valid_tracker,
            self.critic_loss_tracker,
            self.generator_loss_tracker,
            self.value_function_tracker,
            self.chi_rmse_tracker,
            self.critic_grad_norm,
            self.generator_grad_norm
        ] + self.critic_grad_norms + self.generator_grad_norms

# %%
"""