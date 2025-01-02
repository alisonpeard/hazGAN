# %%
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from functools import partial
from keras import optimizers
from keras import ops
import torch
from torch.autograd import grad

# imports for 'fit()' overwrite
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src.utils import traceback_utils
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from torch.utils.data import WeightedRandomSampler
import math

from ..constants import SAMPLE_CONFIG
from .models import Critic
from .models import Generator
from .augment import DiffAugment
from ..statistics.metrics import chi_rmse


class TorchEpochIterator(EpochIterator):
    def _get_iterator(self):
        return self.data_adapter.get_torch_dataloader()


def sample_gumbel(shape, eps=1e-20, temperature=1., offset=0., seed=None):
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


class infInitialisedMean(keras.metrics.Mean):
    def reset_states(self):
        super().reset_states()
        self.update_state(float('inf'))


class WGANGP(keras.Model):
    """Refernece: https://keras.io/guides/custom_train_step_in_torch/"""
    def __init__(self, config, device:str):
        super(WGANGP, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dims']
        self.lambda_gp = config['lambda_gp']
        self.lambda_var = config['lambda_var']
        self.latent_space_distn = setup_latents(config['latent_space_distn'])
        
        if config['augment_policy'] != '':
            raise NotImplementedError(
                "DiffAugment not yet implemented in Pytorch version." +
                " received policy: " + config['augment_policy']
                )
        self.augment = partial(DiffAugment, policy=config['augment_policy'])
        
        self.seed = config['seed']
        self.device = device if getattr(torch, device).is_available() else "cpu"
        self.training_balance = config['training_balance']
        self.nfields = len(config['fields'])

        self.generator = Generator(config, nfields=self.nfields).to(self.device)
        self.critic = Critic(config, nfields=self.nfields).to(self.device)

        self.trainable_vars = [
            *self.generator.parameters(),
            *self.critic.parameters()
        ]

        if config['gumbel']:
            self._uniform = lambda x: torch.exp(-torch.exp(-x))
        else:
            self._uniform = lambda x: x

        # stateful metrics
        self.chi_rmse_tracker = infInitialisedMean(name="chi_rmse")
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

        # image statistcs
        self.fake_mean = keras.metrics.Mean(name="fake_mean")
        self.real_mean = keras.metrics.Mean(name="real_mean")
        self.fake_std = keras.metrics.Mean(name="fake_std")
        self.real_std = keras.metrics.Mean(name="real_std")

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
        super().to(self.device)


    def call(self, label=[1], condition=[1.], nsamples=None,
             noise=None, temp=1., offset=0, seed=None,
             verbose=False):
        '''Return uniformly distributed samples from the generator.'''
        if nsamples is None:
            nsamples = len(label)
        
        if noise is None:
            noise = self.latent_space_distn(
                (nsamples, self.latent_dim),
                temperature=temp,
                offset=offset,
                seed=seed
                )
            
        def _tensor(x):
            if not isinstance(x, torch.Tensor):
                return torch.tensor(x)
            else:
                return x
            
        label1d = _tensor(label).to(dtype=torch.int64, device=self.device).reshape(-1,)
        condition2d = _tensor(condition).to(dtype=torch.float32, device=self.device).reshape(-1,1)
        raw = self.generator(noise, label=label1d, condition=condition2d)
        
        if verbose:
            print("Minimum before transformation:", ops.min(raw))
            print("Maximum before transformation:", ops.max(raw))
        return self._uniform(raw)
    

    def evaluate(self, x:dict, *args, **kwargs) -> dict:
        '''Overwrite evaluation function for custom data.'''
        score_valid = 0.
        with torch.no_grad():
            for n, batch in enumerate(x):
                try:
                    data = batch['uniform']
                    condition = batch["condition"]
                    label = batch["label"]
                    critic_score = self.critic(data, label, condition)
                    score_valid += ops.mean(critic_score)

                except Exception as e:
                    print(e)
                    break

        score_valid = score_valid / (n + 1)
        self.critic_valid_tracker.update_state(score_valid)
        return {'critic': self.critic_valid_tracker.result()}

    def __repr__(self) -> str:
        out = "\nWasserstein GAN with Gradient Penalty\n"
        out += "-------------------------------------\n"
        out += self.config.__repr__()
        return out

    def _gradient_penalty(self, data, fake_data, condition, label):
        eps = keras.random.uniform((data.shape[0], 1, 1, 1))
        differences = fake_data - data
        interpolates = data + (eps * differences)
        score_interpolates = self.critic(interpolates, label, condition)
        gradients = grad(score_interpolates, interpolates,
                         grad_outputs=torch.ones(score_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(data.shape[0], -1)
        gradient_norm = ops.sqrt(ops.sum(gradients ** 2, axis=1) + 1e-12)
        gradient_penalty = self.lambda_gp * ops.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    

    def _variance_penalty(self, data, fake_data):
        """Maximise variance or minimise variance difference."""
        train_batch_var = torch.var(data, dim=0)
        gen_batch_var = torch.var(fake_data, dim=0)
        variance_diff = train_batch_var - gen_batch_var # later
        variance_penalty = torch.sqrt(torch.sum(torch.square(variance_diff)))
        return variance_penalty
    

    def _train_critic(self, data, label, condition, batch_size) -> None:
        noise = self.latent_space_distn((batch_size, self.latent_dim))
        fake_data = self.generator(noise, label, condition)
        score_real = self.critic(data, label, condition)
        score_fake = self.critic(fake_data, label, condition)

        gradient_penalty = self._gradient_penalty(data, fake_data, condition, label)
        variance_penalty = self._variance_penalty(data, fake_data)

        self.zero_grad()
        loss = ops.mean(score_fake) - ops.mean(score_real)
        loss += self.lambda_gp * gradient_penalty
        loss += self.lambda_var * variance_penalty
        loss.backward()

        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.critic_optimizer.apply(grads, self.critic.trainable_weights)
        grad_norms = [ops.norm(grad) for grad in grads]

        self.critic_loss_tracker.update_state(loss)
        self.value_function_tracker.update_state(-loss)
        self.critic_real_tracker.update_state(ops.mean(score_real))
        self.critic_fake_tracker.update_state(ops.mean(score_fake))
        self.gradient_penalty_tracker.update_state(gradient_penalty)
        self.critic_grad_norm.update_state(ops.norm(grad_norms))
        self.critic_steps.update_state(1)

        # update image statistics
        self.fake_mean.update_state(ops.mean(fake_data))
        self.real_mean.update_state(ops.mean(data))
        self.fake_std.update_state(ops.std(fake_data))
        self.real_std.update_state(ops.std(data))


    def _chi_wrapper(self, real, fake):
        """Torch wrapper for chi_rmse."""
        real = real.detach().cpu()
        fake = fake.detach().cpu()
        chi = chi_rmse(real, fake)
        return torch.tensor(chi, dtype=torch.float32)


    def _train_generator(self, data, label, condition, batch_size) -> None:
        noise = self.latent_space_distn((batch_size, self.latent_dim))
        generated_data = self.generator(noise, label, condition)
        critic_score = self.critic(generated_data, label, condition)
        chi = self._chi_wrapper(self._uniform(data), self._uniform(generated_data))

        self.zero_grad()
        loss = -ops.mean(critic_score)
        loss.backward()

        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.generator_optimizer.apply(grads, self.generator.trainable_weights)
        grad_norms = [ops.norm(grad) for grad in grads]

        self.generator_loss_tracker.update_state(loss)
        self.chi_rmse_tracker.update_state(chi)
        self.generator_grad_norm.update_state(ops.norm(grad_norms))
        self.generator_steps.update_state(1)


    def train_step(self, batch:dict) -> dict:
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
            'gradient_penalty': self.gradient_penalty_tracker.result(),
            'fake_mean': self.fake_mean.result(),
            'real_mean': self.real_mean.result(),
            'fake_std': self.fake_std.result(),
            'real_std': self.real_std.result()
        }
        
        if self.critic_steps.result() % self.training_balance == 0:
            self._train_generator(data, label, condition, batch_size)
            metrics['chi_rmse'] = self.chi_rmse_tracker.result()
            metrics["generator_grad_norm"] = self.generator_grad_norm.result()
            metrics['generator_steps'] = self.generator_steps.result()
            metrics["generator_loss"] = self.generator_loss_tracker.result()

        # update metrics
        self.images_seen.update_state(batch_size)
        metrics["critic_grad_norm"] = self.critic_grad_norm.result()
        metrics['critic_steps'] = self.critic_steps.result()
        metrics["images_seen"] = self.images_seen.result()

        print('\n') # uncomment to log metrics on newlines

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
            self.critic_grad_norm,
            self.generator_grad_norm,
            self.fake_mean,
            self.real_mean,
            self.fake_std,
            self.real_std,
        ]

    # - - - - - - - - - BELOW HERE IS A CUSTOM FIT METHOD - - - - - - - - - - - - - - - - - - - - - - - |
    @staticmethod
    def get_initial_weights(labels:torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(labels)
        weights = counts / counts.sum()
        return weights

    @staticmethod
    def linear_weights(
            initial_weights,
            target_weights=torch.tensor([0., 0., 1.]),
            epochs=1
            ) -> callable:
        weight_matrix = torch.empty((epochs, len(initial_weights)))
        for i in range(len(initial_weights)):
            weight_matrix[:, i] = torch.linspace(
                initial_weights[i],
                target_weights[i],
                epochs
            )
        def get_weights(epoch) -> torch.Tensor:
            return weight_matrix[epoch]
        return get_weights
    
    @staticmethod
    def cosine_decay(x, y, steps):
        weights = []
        for step in range(steps):
            weights.append(y + (x - y) * (1 + ops.cos(math.pi * step / steps)) / 2)
        return torch.tensor(weights)

    @staticmethod
    def cosine_weights(
            initial_weights,
            target_weights=torch.tensor([0., 0., 1.]),
            epochs=1
            ) -> callable:
        weight_matrix = torch.empty((epochs, len(initial_weights)))
        for i in range(len(initial_weights)):
            weight_matrix[:, i] = WGANGP.cosine_decay(initial_weights[i], target_weights[i], epochs)
        def get_weights(epoch) -> torch.Tensor:
            return weight_matrix[epoch]
        return get_weights

    @staticmethod
    def update_dataloader_weights(dataloader:WeightedRandomSampler, weights) -> WeightedRandomSampler:
        dataloader.weights = weights
        return dataloader

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        target_weights=torch.tensor([0., .5, .5]),
    ):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )

        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            # TODO: Support torch tensors for validation data.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        update_epoch_iterator = partial(
            TorchEpochIterator,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution
            )

        target_weights = target_weights
        initial_weights = self.get_initial_weights(x.dataset.data['label'])
        weight_iterator = self.cosine_weights(initial_weights, target_weights, epochs)

        def update_dataloader(x, epoch, weight_iterator) -> TorchEpochIterator:
            weights = weight_iterator(epoch)
            x = self.update_dataloader_weights(x, weights)
            return x, weights
        
        x = self.update_dataloader_weights(x, initial_weights)
        weights = initial_weights
        epoch_iterator = update_epoch_iterator(x)
        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):

            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            # Switch the torch Module to training mode. Inform torch layers to
            # do training behavior in case the user did not use `self.training`
            # when implementing a custom layer with torch layers.
            self.train()

            logs = {}
            for step, data in epoch_iterator:
                # Callbacks
                callbacks.on_train_batch_begin(step)

                logs = self.train_function(data)
                for i, weight in enumerate(weights):
                    logs[f"weight_{i}"] = weight

                # Callbacks
                callbacks.on_train_batch_end(step, logs)
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Switch the torch Module back to testing mode.
            self.eval()

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create TorchEpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TorchEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)

            # Update dataloader to new resampling weights
            x, weights = update_dataloader(x, epoch, weight_iterator)
            epoch_iterator = update_epoch_iterator(x)
            self._symbolic_build(iterator=epoch_iterator)
            epoch_iterator.reset()

            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    # - - - - - - - - - END OF CUSTOM FIT METHOD - - - - - - - - - - - - - - - - - - - - - - - |
# %%