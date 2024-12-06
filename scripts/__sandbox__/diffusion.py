"""
06-12-2024 Trying diffusion model from denoising_diffusion_pytorch
https://github.com/lucidrains/denoising-diffusion-pytorch
"""

# %%
# last run in wgan-pytorch 06-12-2024
import os
# >>> !python -m pip install denoising_diffusion_pytorch

# %%
import torch
import torchvision
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
# torch.mps.set_per_process_memory_fraction(0.0) 
device = torch.device("mps")

model = Unet(
    dim=64,
    channels=1,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000    # number of steps, default=1000
)


# training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
datadir = '/Users/alison/Documents/DPhil/paper1.nosync/training/18x22'
training_images = np.load(os.path.join(datadir, "data_filtered_60000.npz"))['x']
training_images = training_images[:1000, ...]
training_images = training_images.transpose(0, 3, 1, 2)
training_images = torch.tensor(training_images).float()
training_images = torchvision.transforms.Resize((32, 32))(training_images)

# visualise one image
import matplotlib.pyplot as plt
plt.imshow(training_images[0].permute(1, 2, 0).numpy().squeeze(), cmap='Spectral_r')

# %%
# training_images = training_images.to(device)
# model = model.to(device)
# diffusion = diffusion.to(device)

loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size=4)
sampled_images.shape # (4, 3, 128, 128)

# %%
sampled_images = sampled_images.cpu()
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
for i, ax in enumerate(axs.flat):
    ax.imshow(sampled_images[i].permute(1, 2, 0).numpy().squeeze(), cmap='Spectral_r')
    ax.axis('off')
    ax.set_title('')

# %%