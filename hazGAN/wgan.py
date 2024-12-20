# %%
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from functools import partial
from keras import optimizers, ops

import torch
from torch.autograd import grad


from constants import SAMPLE_CONFIG
from torch_utils.models import Critic, Generator
from torch_utils.augment import DiffAugment
from stat_utils import chi_rmse


def sample_gumbel(shape, eps=1e-20, temperature=1., offset=0., seed=None, device='mps'):
    """Sample from Gumbel(0, 1)"""
    O = offset
    T = temperature
    U = keras.random.uniform(shape, minval=0, maxval=1, seed=seed)
    return O - T * ops.log(-ops.log(U + eps) + eps)


def setup_latents(distribution:str):
    """Allows me to specify distribution in wandb sweeps."""
    assert distribution in ['gumbel', 'uniform', 'normal'], "Invalid distribution specified."
    keras.random.gumbel = sample_gumbel
    return getattr(keras.random, distribution)


class WGANGP(keras.Model):
    """Refernece: https://keras.io/guides/custom_train_step_in_torch/"""
    def __init__(self, config, nfields=2):
        super(WGANGP, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.lambda_gp = config['lambda_gp']
        self.latent_space_distn = setup_latents(config['latent_space_distn'])
        self.augment = partial(DiffAugment, policy=config['augment_policy'])
        self.seed = config['seed']
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.training_balance = config['training_balance']
        self.generator = Generator(config, nfields=nfields).to(self.device)
        self.critic = Critic(config, nfields=nfields).to(self.device)

        self.trainable_vars = [
            *self.generator.parameters(),
            *self.critic.parameters()
        ]

        if config['gumbel']:
            self.inv = lambda x: -ops.log(-ops.log(x))
        else:
            self.inv = lambda x: x

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

        self.built = True

    def compile(self, *args, **kwargs) -> None:
        super().compile(*args, **kwargs)
        config = self.config
        optimizer = getattr(optimizers, config['optimizer'])
        opt_kwargs = {'learning_rate': config['learning_rate'],
                  'beta_1': config['beta_1'],
                  'beta_2': config['beta_2']}
        self.critic_optimizer = optimizer(**opt_kwargs)
        self.generator_optimizer = optimizer(**opt_kwargs)


    def call(self, label, condition, nsamples=5,
             noise=None, temp=1., offset=0, seed=None,
             verbose=False):
        '''Return uniformly distributed samples from the generator.'''
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

        raw = self.generator(noise, label, condition)
        if verbose:
            print("Minimum before transformation:", ops.min(raw))
            print("Maximum before transformation:", ops.max(raw))
        return self.inv(raw)
    

    def evaluate(self, x) -> dict:
        '''Overwrite evaluation function for custom data.'''
        score_valid = 0
        for n, batch in enumerate(x):
            try:
                data = batch['uniform']
                condition = batch["condition"]
                label = batch["label"]
                critic_score = self.critic([data, condition, label], training=False)
                score_valid += ops.mean(critic_score)
            except Exception as e:
                print(e)
                break
        score_valid = score_valid / (n + 1)
        self.critic_valid_tracker.update_state(score_valid)
        return {'critic': self.critic_valid_tracker.result()}
    

    def _gradient_penalty(self, data, fake_data, condition, label):
        eps = keras.random.uniform((data.shape[0], 1, 1, 1))
        differences = fake_data - data
        interpolates = data + (eps * differences)
        score_interpolates = self.critic(interpolates, label, condition)
        gradients = grad(score_interpolates, interpolates,
                         create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(data.shape[0], -1)
        gradient_norm = ops.sqrt(ops.sum(gradients ** 2, axis=1) + 1e-12)
        gradient_penalty = self.lambda_gp * ops.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    

    def _train_critic(self, data, label, condition, batch_size) -> None:
        noise = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator(noise, label, condition)
        score_real = self.critic(data, label, condition)
        score_fake = self.critic(fake_data, label, condition)

        gradient_penalty = self._gradient_penalty(data, fake_data, condition, label)

        self.zero_grad()
        loss = ops.mean(score_fake) - ops.mean(score_real)
        loss += self.lambda_gp * gradient_penalty
        loss.backward()

        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.critic_optimizer.apply(grads, self.critic.trainable_weights)
        grad_norms = [ops.norm(grad) for grad in grads]

        self.critic_loss_tracker(loss)
        self.value_function_tracker.update_state(-loss)
        self.critic_real_tracker.update_state(ops.mean(score_real))
        self.critic_fake_tracker.update_state(ops.mean(score_fake))
        self.gradient_penalty_tracker.update_state(gradient_penalty)
        self.critic_grad_norm.update_state(ops.norm(grad_norms))
        self.critic_steps.update_state(1)


    def _train_generator(self, data, label, condition, batch_size) -> None:
        noise = self.latent_space_distn((batch_size, self.latent_dim))
        generated_data = self.generator(noise, label, condition)
        critic_score = self.critic(generated_data, label, condition)
        chi_rmse = chi_rmse(self.inv(data), self.inv(generated_data))

        self.zero_grad()
        loss = -ops.mean(critic_score)
        loss.backward()

        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.generator_optimizer.apply(grads, self.generator.trainable_weights)
        grad_norms = [ops.norm(grad) for grad in grads]

        self.generator_loss_tracker(loss)
        self.chi_rmse_tracker(chi_rmse)
        self.generator_grad_norm.update_state(ops.norm(grad_norms))
        self.generator_steps.update_state(1)


    def train_step(self, batch) -> dict:
        data = batch['uniform']
        condition = batch['condition']
        label = batch['label']
        batch_size = data.shape[0]
        
        self._train_critic(data, label, condition, batch_size)

        metrics = {
            "critic_loss": self.critic_loss_tracker.result(),
            "value_function": -self.value_function_tracker.result(),
            'critic_real': self.critic_real_tracker.result(),
            'critic_fake': self.critic_fake_tracker.result(),
            'gradient_penalty': self.gradient_penalty_tracker.result()
        }
        
        if self.critic_steps.result() % self.training_balance == 0:
            self._train_generator(data, label, condition, batch_size)

        # update metrics
        self.images_seen.update_state(batch_size)
        metrics["generator_loss"] = self.generator_loss_tracker.result()
        metrics['critic_steps'] = self.critic_steps.result()
        metrics['generator_steps'] = self.generator_steps.result()
        metrics["images_seen"] = self.images_seen.result()

        metrics['chi_rmse'] = self.chi_rmse_tracker.result()
        metrics["critic_grad_norm"] = self.critic_grad_norm.result()
        metrics["generator_grad_norm"] = self.generator_grad_norm.result()
        
        return metrics

    @property
    def metrics(self) -> list:
        '''Define which stateful metrics to reset per-epoch.'''
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
        ]
    


if __name__ == "__main__":
    import torch

    model = WGANGP(SAMPLE_CONFIG)
    device = 'mps'
    z = torch.rand(1, SAMPLE_CONFIG['latent_dim'], device=device)
    label = torch.randint(0, SAMPLE_CONFIG['nconditions'], (1,), device=device)
    condition = torch.rand(1, 1, device=device)
    x = torch.rand(1, 2, 20, 24, device=device)
    model.compile()

    batch = {'uniform': x, 'condition': condition, 'label': label}
    model.fit(batch, epochs=1, verbose=1)


# %%