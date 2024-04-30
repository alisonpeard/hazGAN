# %%
import os
import numpy as np
import wandb
import pandas as pd
from scipy.stats import ksone # Kolmogorov-Smirnov test
import tensorflow as tf
import tensorflow_probability as tfp
import hazGAN as hg
import matplotlib.pyplot as plt
import wandb

def ks_critical_value(n_trials, alpha):
    return ksone.ppf(1 - alpha / 2, n_trials)

plt.rcParams["font.family"] = "monospace"

plot_kwargs = {"dpi": 500, "bbox_inches": "tight"}
hist_kwargs = {"density": True, "color": "lightgrey", "alpha": 0.6, "edgecolor": "k"}

# %%
wd = "/Users/alison/Documents/DPhil/multivariate/hazGAN"
RUNNAME = "_240430-gumbel"
os.chdir(os.path.join(wd, "saved-models", RUNNAME))
paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
cmaps = ["YlOrRd", "PuBu", "YlGnBu"]
figdir = "/Users/alison/Documents/DPhil/multivariate/hazGAN/figures/results"
# %%
wandb.init(project="test", mode="disabled")
config = wandb.config
wgan = hg.WGAN(wandb.config, nchannels=2)
wgan.generator.load_weights(os.path.join(wd, "saved-models", RUNNAME, f"generator.weights.h5"))
wgan.generator.summary()
# %% 
if config.gumbel:
    # is it generating gumbel-distributed marginals?
    from scipy.stats import gumbel_r
    n = 1000
    i, j, c = np.random.randint(18), np.random.randint(22), 0  # choose random pixel
    y_ij = wgan.generator(tf.random.normal((n, 100))).numpy()[:, i, j, c]
    u_ij = wgan(nsamples=n).numpy()[:, i, j, c]

    # compare generated to standard gumbel distribution
    u = np.linspace(gumbel_r.ppf(1e-6), gumbel_r.ppf(1-1e-6), n)
    y = gumbel_r.pdf(u)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    gumbel = gum = tfp.distributions.Gumbel(0, 1)
    axs[0].hist(y_ij, bins=50, **hist_kwargs)
    axs[0].plot(u, y, "--r", label='Gumbel(0,1)')
    axs[0].legend()
    axs[0].set_title("Generated data")

    axs[1].hist(u_ij, bins=50, **hist_kwargs)
    axs[1].set_title("Uniform data")

# %% initialise model
fake_u = hg.unpad(wgan(nsamples=1000), paddings).numpy()
[train_u, test_u], [train_x, test_x], [train_m, test_m], [train_z, test_z], params = hg.load_training(
    "/Users/alison/Documents/DPhil/multivariate/era5_data",
    config.train_size,
    padding_mode=None,
    gumbel_marginals=False,
    numpy=True
)

# %% plot 100 images
fig = hg.plot_one_hundred_images(fake_u, cmap="Spectral_r", vmin=0, vmax=1)
fig = hg.plot_one_hundred_images(test_u, cmap="Spectral_r", suptitle="Test marginals", vmin=0, vmax=1)
# %%
# Pixel distribution histograms
i, j, c = np.random.randint(18), np.random.randint(22), 0  # choose random pixel

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.hist(
    fake_u[..., i, j, c].ravel(),
    bins=50,
    label="Generated",
    **hist_kwargs,
)
# %% Look at total distribution of uniform pixels
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].hist(train_u.ravel(), bins=50, **hist_kwargs)
axs[0].set_title("Train data")
axs[1].hist(test_u.ravel(), bins=50, **hist_kwargs)
axs[1].set_title("Test data")
axs[2].hist(fake_u.ravel(), bins=50, **hist_kwargs)
axs[2].set_title("Generated data")

# %% Kolmogorov-Smirnov test for uniformity
channel = 0
alpha = 0.05
baseline = np.random.uniform(0, 1, (1000, 18, 22, 1)).astype(np.float32)

ks_stat_base = hg.kstest_loss(baseline).numpy().squeeze()
ks_stat_test = hg.kstest_loss(test_u[..., [channel]]).numpy().squeeze()
ks_stat_train = hg.kstest_loss(train_u[..., [channel]]).numpy().squeeze()
ks_stat_fake = hg.kstest_loss(fake_u[..., [channel]]).numpy().squeeze()

ks_crit_base = ks_critical_value(baseline.shape[0], alpha)
ks_crit_test = ks_critical_value(test_u.shape[0], alpha)
ks_crit_train = ks_critical_value(train_u.shape[0], alpha)
ks_crit_fake = ks_critical_value(fake_u.shape[0], alpha)

k_reject_base = ks_stat_base > ks_crit_base
k_reject_test = ks_stat_test > ks_crit_test
k_reject_train = ks_stat_train > ks_crit_train
k_reject_fake = ks_stat_fake > ks_crit_fake

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
im = axs[0].imshow(np.where(ks_stat_base <= ks_crit_base, 0, 1), cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(im)
im = axs[1].imshow(np.where(ks_stat_test <= ks_crit_test, 0, 1), cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(im)
im = axs[2].imshow(np.where(ks_stat_train <= ks_crit_train, 0, 1), cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(im)
im = axs[3].imshow(np.where(ks_stat_fake <= ks_crit_fake, 0, 1), cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(im)
fig.suptitle("KS test for uniformity, reject null hypothesis where 1")

axs[0].set_title("Baseline")
axs[1].set_title("Test data")
axs[2].set_title("Train data")
axs[3].set_title("Generated data")

print("Number of base rejections: ", k_reject_base.sum())
print("Number of test rejections: ", k_reject_test.sum())
print("Number of train rejections: ", k_reject_train.sum())
print("Number of fake rejections: ", k_reject_fake.sum())
# p-value

# %% Q-Q and Gumbel-Gumbel plots
i, j, c = np.random.randint(18), np.random.randint(22), 0  # choose random pixel

y = sorted(fake_u[..., i, j, c].ravel())
x = np.linspace(min(y), max(y), len(y))

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(x, y, "-ok", markersize=1)
axs[0].plot(x, x, "--r")
axs[0].set_title("Uniform P-P plot")

axs[0].plot(x, y, "-ok", markersize=1)
axs[0].plot(x, x, "--r")
axs[0].set_title("Uniform P-P plot")

y_gumbel = -np.log(-np.log(y))
x_gumbel = -np.log(-np.log(x))

axs[1].plot(x_gumbel, y_gumbel, "-ok", markersize=1)
axs[1].plot(x_gumbel, x_gumbel, "--r")
axs[1].set_title("Gumbel Q-Q plot")

for ax in axs:
    ax.set_xlabel("Uniform")
    ax.set_ylabel("Generated data")
    ax.label_outer()

fig.suptitle(f"Data distributions, pixel ({i}, {j}), channel {c}\n")

# Pixel relationships
# --------------------------------------------------------------------------------------------
# %% calculate Pearson correlations
channel = 1
corrs_train = hg.get_all_correlations(train_u[..., channel])
corrs_test = hg.get_all_correlations(test_u[..., channel])
corrs_u = hg.get_all_correlations(fake_u[..., channel])
# %% plot correlation results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(corrs_train, cmap="coolwarm", vmin=0, vmax=1)
axs[1].imshow(corrs_test, cmap="coolwarm", vmin=0, vmax=1)
im = axs[2].imshow(corrs_u, cmap="coolwarm", vmin=0, vmax=1)
axs[0].set_title("Train data")
axs[1].set_title("Test data")
axs[2].set_title("Generated data")
fig.suptitle(r"Pairwise Pearson correlation $\rho$ for channel {}".format(channel))
fig.colorbar(im, ax=axs, orientation="vertical", shrink=0.8, aspect=20)

# %% extremal correlations across space for generated data
channel = 0
ecs_fake = hg.pairwise_extremal_coeffs(fake_u[..., channel]).numpy()
ecs_train = hg.pairwise_extremal_coeffs(train_u[..., channel]).numpy()
ecs_test = hg.pairwise_extremal_coeffs(test_u[..., channel]).numpy()

rmse_chi_train_test = np.sqrt(np.nanmean((ecs_train - ecs_test) ** 2))
rmse_chi_train = np.sqrt(np.nanmean((ecs_fake - ecs_train) ** 2))
rmse_chi_test = np.sqrt(np.nanmean((ecs_fake - ecs_test) ** 2))

print("RMSE chi train-test: ", rmse_chi_train_test)
print("RMSE chi train: ", rmse_chi_train)
print("RMSE chi test: ", rmse_chi_test)
# %%
vmin = min(ecs_fake.min(), ecs_train.min(), ecs_test.min())
vmax = max(ecs_fake.max(), ecs_train.max(), ecs_test.max())

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
im = axs[0].imshow(ecs_train, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
im = axs[1].imshow(ecs_test, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
im = axs[2].imshow(ecs_fake, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=axs, orientation="vertical", shrink=0.8, aspect=40)
axs[0].set_title("Train data")
axs[1].set_title("Test data")
axs[2].set_title("Generated data")
fig.suptitle(r"Pairwise $\hat \theta$ for channel {}".format(channel))
# %%
# plot the differences in extremal coefficients
diffs = abs(ecs_fake - ecs_test)
vmin = diffs.min()
vmax = diffs.max()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(diffs, cmap="coolwarm", vmin=vmin, vmax=vmax)
fig.colorbar(im)
ax.set_title("Difference in extremal coefficients")
# %%
