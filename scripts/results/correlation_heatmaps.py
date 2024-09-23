"""
Make correlation/extremal correlation heatmaps and compare them across the training, test and generated data.
"""
#%%
import os 
import numpy as np
import yaml
import xarray as xr
import hazGAN as hg
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams['font.family'] = 'serif'

def open_config(runname, dir):
    configfile = open(os.path.join(dir, runname, "config-defaults.yaml"), "r")
    config = yaml.load(configfile, Loader=yaml.FullLoader)
    config = {key: value["value"] for key, value in config.items()}
    return config

# %%
res = (18, 22)
RUNNAME = "colorful-sweep-2" #"amber-sweep-13" 
datadir = f'/Users/alison/Documents/DPhil/paper1.nosync/training/{res[0]}x{res[1]}'
samplesdir = f'/Users/alison/Documents/DPhil/paper1.nosync/samples'
config = open_config(RUNNAME, "/Users/alison/Documents/DPhil/paper1.nosync/hazGAN/saved-models")
data = xr.open_dataset(os.path.join(datadir, "data.nc")).sel(channel=['u10', 'tp'])
samples_ds = xr.open_dataset(os.path.join(samplesdir, f"{RUNNAME}.nc"))
occurence_rate = 18.033 # from R 
ntrain = config['train_size']
# samples_ds = samples_ds.isel(sample=slice(0, ntrain)) # maybe fairer?
train_ds = data.isel(time=slice(-ntrain, None))
test_ds = data.isel(time=slice(0, -ntrain))
samples_ds = samples_ds.rename({'sample': 'time'})

# %% ------Cross-channel correlations------
train = train_ds.uniform.values
test = test_ds.uniform.values
samples = samples_ds.uniform.values

def get_channel_corrs(x, channel0, channel1):
    n, h, w, c = x.shape
    
    corrs = np.ones([h, w])
    for i in range(h):
        for j in range(w):
            corr = np.corrcoef(x[:, i, j, channel0], x[:, i, j, channel1])[0, 1]
            corrs[i, j] = corr
    return corrs

channels = [0, 1]
for c0, c1 in combinations(channels, 2):
    train_corrs = get_channel_corrs(train, c0, c1)
    test_corrs = get_channel_corrs(test, c0, c1)
    gan_corrs = get_channel_corrs(samples, c0, c1)
    
    mask = np.ma.masked_invalid(train_corrs)
    gan_corrs[mask.mask] = np.nan

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5),
                       gridspec_kw={
                           'wspace': .02,
                           'width_ratios': [1, 1, 1, .05]}
                           )
    im = ax[0].imshow(train_corrs, vmin=0, vmax=1, cmap="coolwarm")
    ax[1].imshow(test_corrs, vmin=0, vmax=1, cmap="coolwarm")
    ax[2].imshow(gan_corrs, vmin=0, vmax=1, cmap="coolwarm")

    ax[0].set_title("Train", fontsize=16)
    ax[1].set_title("Test", fontsize=16)
    ax[2].set_title("hazGAN", fontsize=16)
    
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.invert_yaxis()
    hg.add_watermark(a, RUNNAME)

    fig.colorbar(im, cax=ax[3], extend='both', orientation='vertical')
    # fig.suptitle("Correlation between {} and {}".format(hg.channel_labels[c0], hg.channel_labels[c1]))
    ax[0].set_ylabel('Correlation', fontsize=18);


# %% ------Cross-channel extremal correlations------
# now do the same for extremals (but in 3d because you can!)
def get_channel_ext_coefs(x):
    n, h, w, c = x.shape
    excoefs = hg.get_extremal_coeffs_nd(x, [*range(h * w)])
    excoefs = np.array([*excoefs.values()]).reshape(h, w)
    return excoefs

# correlations across all variables
excoefs_train = get_channel_ext_coefs(train)
excoefs_test = get_channel_ext_coefs(test)
excoefs_gan = get_channel_ext_coefs(samples)

fig, ax = plt.subplots(1, 4, figsize=(12, 3.5),
                       gridspec_kw={
                           'wspace': .02,
                           'width_ratios': [1, 1, 1, .05]}
                           )
cmap = plt.cm.coolwarm_r
cmap.set_over('lightgrey') #cmap(.99)
cmap.set_under(cmap(0))
cmap_kws = {'cmap': cmap, 'vmin': 1, 'vmax': 2.5}

im = ax[0].imshow(excoefs_train, **cmap_kws)
im = ax[1].imshow(excoefs_test, **cmap_kws)
im = ax[2].imshow(excoefs_gan, **cmap_kws)

for a in ax:
    a.set_yticks([])
    a.set_xticks([])
    a.invert_yaxis()
hg.add_watermark(a, RUNNAME)
    
ax[0].set_title('Train', fontsize=16)
ax[1].set_title('Test', fontsize=16)
ax[2].set_title('hazGAN', fontsize=16);

fig.colorbar(im, cax=ax[3], extend='both', orientation='vertical')
ax[0].set_ylabel(r'Extremal coeff.', fontsize=18);
# fig.suptitle(r'$\hat \theta$ between channels');

rmse_chi_train_test = np.sqrt(np.nanmean((excoefs_train - excoefs_test) ** 2))
rmse_chi_train = np.sqrt(np.nanmean((excoefs_gan - excoefs_train) ** 2))
rmse_chi_test = np.sqrt(np.nanmean((excoefs_gan - excoefs_test) ** 2))

print(f"RMSE chi train-test: {rmse_chi_train_test:.4f}")
print(f"RMSE chi train: {rmse_chi_train:.4f}")
print(f"RMSE chi test: {rmse_chi_test:.4f}")

# %% -----Spatial correlations------
channel = 0
train = train_ds.isel(channel=channel).uniform.values
test = test_ds.isel(channel=channel).uniform.values
samples = samples_ds.isel(channel=channel).uniform.values

corr_train = hg.get_all_correlations(train)
corr_test = hg.get_all_correlations(test)
corr_gen = hg.get_all_correlations(samples)

fig, axs = plt.subplots(1, 4, figsize=(12, 3.5),
                       gridspec_kw={
                           'wspace': .02,
                           'width_ratios': [1, 1, 1, .05]}
                           )

axs[0].imshow(corr_train, cmap="coolwarm", vmin=0, vmax=1)
axs[1].imshow(corr_test, cmap="coolwarm", vmin=0, vmax=1)
im = axs[2].imshow(corr_gen, cmap="coolwarm", vmin=0, vmax=1)
axs[0].set_title("Train", fontsize=16)
axs[1].set_title("Test", fontsize=16)
axs[2].set_title("hazGAN", fontsize=16)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, cax=axs[3], extend='both', orientation='vertical')
axs[0].set_ylabel(r'Correlation', fontsize=18);

# %% ------Extremal Spatial correlations------
ecs_train = hg.pairwise_extremal_coeffs(train.astype(np.float32)).numpy()
ecs_test = hg.pairwise_extremal_coeffs(test.astype(np.float32)).numpy()
ecs_gen = hg.pairwise_extremal_coeffs(samples.astype(np.float32)).numpy()

rmse_chi_train_test = np.sqrt(np.nanmean((ecs_train - ecs_test) ** 2))
rmse_chi_train = np.sqrt(np.nanmean((ecs_gen - ecs_train) ** 2))
rmse_chi_test = np.sqrt(np.nanmean((ecs_gen - ecs_test) ** 2))

print(f"RMSE chi train-test: {rmse_chi_train_test:.4f}")
print(f"RMSE chi train: {rmse_chi_train:.4f}")
print(f"RMSE chi test: {rmse_chi_test:.4f}")

vmin = 1 # min(ecs_gen.min(), ecs_train.min(), ecs_test.min())
# vmax = max(ecs_gen.max(), ecs_train.max(), ecs_test.max())
vmin, vmax = 1, 2.5

cmap = plt.cm.coolwarm_r
cmap.set_over(cmap(.99))
cmap_kws = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}

fig, axs = plt.subplots(1, 4, figsize=(12, 3.5),
                       gridspec_kw={
                           'wspace': .02,
                           'width_ratios': [1, 1, 1, .05]}
                           )

im = axs[0].imshow(ecs_train, **cmap_kws)
im = axs[1].imshow(ecs_test, **cmap_kws)
im = axs[2].imshow(ecs_gen, **cmap_kws)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, cax=axs[3], extend='both', orientation='vertical');

axs[0].set_title("Train", fontsize=16)
axs[1].set_title("Test", fontsize=16)
axs[2].set_title("hazGAN", fontsize=16)
axs[0].set_ylabel('Extremal coeff.', fontsize=18);

# look at differences
diffs = abs(ecs_gen - ecs_test)
vmin = diffs.min()
vmax = diffs.max()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(diffs, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
fig.colorbar(im)
# ax.set_title(r"$\hat\theta_{\text{GAN}}-\hat\theta_{\text{test}}$");
# %%
