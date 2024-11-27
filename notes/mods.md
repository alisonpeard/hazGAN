# Modifications

## Performance
* Dataset caching?

## EMA and learning rates
> Implemented, makes training slower so unsure of effect
* Higher EMA momentum => more smoothing, 0.99 is considered low,
    * 0.999 - 9.9999
* If not too computationally expensive, update ever 1-2 batches
* LR of 1e-5 might be too conservative and can lead to slow convergence, weak grads and undertraining and too low to escape local minima (makes sense)
    * 1e-4, 3e-4, or even 1e-3
    * Option to have critic LR slightly lower than generator, e.g. (G;1e-4, C:3e-5 or 1e-4)

## Batch sizes and number of epochs etc
GANs have a more fluid concept of epochs compared to supervised learning, since they usually use an infinite (`.repear()`) dataset.


## Orthogonal weight initialization
> Implemented, unsure of effect

A matrix $W$ is orthogonal iff $$W^\top W=WW^\top = I. $$ Hence, $$||Wx||^2_2 = (Wx)^\top Wx = x^\top W^\top W x = x^\top x = ||x||^2_2.$$
So $W$ will preserve both the input norm of $x$ and the input gradient norm.
```
# If W is orthogonal:
||W * x|| = ||x||  # Preserves input norm
||W^T * g|| = ||g|| # Preserves gradient norm
```

Implemented using `wrappers.py`.
```
# For BatchNorm:
gamma_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)
beta_initializer = tf.keras.initializers.Zeros()
```
Also try initialising weights to N(0,002) or orthogonal limit


## Gradient norm monitoring
> Not implemented
Monitor gradient norms during training to detect instability early:
```
def compute_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# During training:
c_grad_norm = compute_grad_norm(critic)
g_grad_norm = compute_grad_norm(generator)
```

## Instance noise
> Not implemented
Instance noise at the start of training and gradually decrease over training. 
This gives the generator a better chance by introducing some overlap between the real and fake distributions.
```
noise_std = max(0, 0.1 * (1 - epoch/50))  # Linear decay over 50 epochs def add_instance_noise(x): return x +torch.randn_like(x) * noise_std
```

## [Spectral normalisation](https://proceedings.neurips.cc/paper/2021/file/4ffb0d2ba92f664c2281970110a2e071-Paper.pdf)
> Not implemented
Additional stability.

## Layer normalisation (critic option, not needed in Gulrajani)
> Not implemented


## Check batch norm momentum
> Not implemented


## Gradient clipping
> Not implemented, possible conflict with gradient penalty?
Gradient clipping if things are exploding

## Minibatch discrimination
> Not relevant, conflicts with gradient penalty
* Discriminator sees entire batch at once rather than single sample
* Models distance between each sample and the rest of the batch
* It then combines these distances with the sample and passes them through the discriminator, giving it extra information
* Requires a smaller batch size and becomes more sensitive to batch size choice

**Steps**:
- Take output of intermediate critic layer
- Multiply by a 3D tensor (i.e., to embed in 2D?)
- Compite $\ell_1$ distance between tows in this matrix and all batch samples
- negative exponential of distances
- Sum to create features for each sample
- Concatenate original input with minibatch features and pass through rest of network



> [Suggestions from Claude](https://claude.ai/chat/95715d9a-ab72-4d18-bd58-425a45b23f06) 



