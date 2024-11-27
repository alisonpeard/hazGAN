import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from .defaults import channel_labels
from .extreme_value_theory import (
    POT,
    get_extremal_coeffs_nd,
    pairwise_extremal_coeffs,
    # interpolate_thresholds, # old, might not need
    # get_extremal_coeffs     # old, might not need
)

# plots for training script
def log_image_to_wandb(fig, name:str, dir:str, **kwargs):
    import wandb
    if wandb.run is not None:
        impath = os.path.join(dir, f"{name}.png")
        fig.savefig(impath, **kwargs)
        wandb.log({name: wandb.Image(impath)})
    else:
        print("Not logging figure, wandb not intialised.")


def figure_one(fake_u:np.array, train_u:np.array, valid_u:np.array, imdir:str) -> None:
    """Plot cross-channel extremal coefficients."""
    def get_channel_ext_coefs(x):
            _, h, w, _ = x.shape
            excoefs = get_extremal_coeffs_nd(x, [*range(h * w)])
            excoefs = np.array([*excoefs.values()]).reshape(h, w)
            return excoefs
    
    excoefs_train = get_channel_ext_coefs(train_u)
    excoefs_valid = get_channel_ext_coefs(valid_u)
    excoefs_fake = get_channel_ext_coefs(fake_u)

    cmap = plt.cm.coolwarm_r
    vmin = 1
    vmax = 2
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1, 1, 1, .05]}
                            )
    im = ax[0].imshow(excoefs_train, vmin=vmin, vmax=vmax, cmap=cmap)
    im = ax[1].imshow(excoefs_valid, vmin=vmin, vmax=vmax, cmap=cmap)
    im = ax[2].imshow(excoefs_fake, vmin=vmin, vmax=vmax, cmap=cmap)

    for a in ax:
        a.set_yticks([])
        a.set_xticks([])
        a.invert_yaxis()
    
    ax[0].set_title('Train', fontsize=16)
    ax[1].set_title('Test', fontsize=16)
    ax[2].set_title('hazGAN', fontsize=16);
    fig.colorbar(im, cax=ax[3], extend='both', orientation='vertical')
    ax[0].set_ylabel('Extremal coeff', fontsize=18);
    log_image_to_wandb(fig, f"extremal_dependence", imdir)


def figure_two(fake_u:np.array, train_u:np.array, valid_u:np.array, imdir:str, channel=0) -> None:
    """Plot spatial extremal coefficients."""
    ecs_train = pairwise_extremal_coeffs(train_u.astype(np.float32)[..., channel]).numpy()
    ecs_valid = pairwise_extremal_coeffs(valid_u.astype(np.float32)[..., channel]).numpy()
    ecs_fake = pairwise_extremal_coeffs(fake_u.astype(np.float32)[..., channel]).numpy()
    
    cmap = plt.cm.coolwarm_r
    vmin = 1
    vmax = 2
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))
    
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1, 1, 1, .05]}
                            )

    im = axs[0].imshow(ecs_train, vmin=vmin, vmax=vmax, cmap=cmap)
    im = axs[1].imshow(ecs_valid, vmin=vmin, vmax=vmax, cmap=cmap)
    im = axs[2].imshow(ecs_fake, vmin=vmin, vmax=vmax, cmap=cmap)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im, cax=axs[3], extend='both', orientation='vertical');
    axs[0].set_title("Train", fontsize=16)
    axs[1].set_title("Test", fontsize=16)
    axs[2].set_title("hazGAN", fontsize=16)
    axs[0].set_ylabel('Extremal coeff.', fontsize=18);

    log_image_to_wandb(fig, f"spatial_dependence", imdir)


def figure_three(fake_u:np.array, train_u:np.array, imdir:str, channel=0,
                 cmap="Spectral_r", levels=20) -> None:
    """Plot the 32 most extreme train and generated percentiles."""
    # prep data to plot
    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    fake = fake_u[..., channel]
    real = train_u[..., channel]

    if fake.shape[0] < 32:
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )

    fake_maxima = np.max(fake, axis=(1, 2))
    fake_sorting = np.argsort(fake_maxima)[::-1]
    fake = fake[fake_sorting, ...]

    real_maxima = np.max(real, axis=(1, 2))
    real_sorting = np.argsort(real_maxima)[::-1]
    real = real[real_sorting, ...]

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plot specs
    fig = plt.figure(figsize=(16, 16), layout="tight")
    subfigs = fig.subfigures(2, 1, hspace=0.2)

    for subfig, item in zip(subfigs, samples.items()):
        axs = subfig.subplots(4, 8, sharex=True, sharey=True,
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
        label = item[0]
        sample = item[1]
        vmin = sample.min()
        vmax = sample.max()
        for i, ax in enumerate(axs.flat):
            im = ax.contourf(lon, lat, sample[i, ...],
                             vmin=vmin, vmax=vmax,
                             cmap=cmap, levels=levels)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Percentiles')
    log_image_to_wandb(fig, f"max_percentiles", imdir)


def figure_four(fake_u, train_u, train_x, params, imdir:str,
                channel=0, cmap="Spectral_r", levels=20) -> None:
    """Plot the 32 most extreme train and generated percentiles."""
    # prep data to plot
    fake = POT.inv_probability_integral_transform(fake_u, train_u, train_x, params)
    real = POT.inv_probability_integral_transform(train_u, train_u, train_x, params)
    fake = fake[..., channel]
    real = train_x[..., channel]

    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    if fake.shape[0] < 32:
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )

    fake_maxima = np.max(fake, axis=(1, 2))
    fake_sorting = np.argsort(fake_maxima)[::-1]
    fake = fake[fake_sorting, ...]

    real_maxima = np.max(real, axis=(1, 2))
    real_sorting = np.argsort(real_maxima)[::-1]
    real = real[real_sorting, ...]

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plot specs
    fig = plt.figure(figsize=(16, 16), layout="tight")
    subfigs = fig.subfigures(2, 1, hspace=0.2)

    for subfig, item in zip(subfigs, samples.items()):
        axs = subfig.subplots(4, 8, sharex=True, sharey=True,
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
        label = item[0]
        sample = item[1]
        vmin = sample.min()
        vmax = sample.max()
        for i, ax in enumerate(axs.flat):
            im = ax.contourf(lon, lat, sample[i, ...],
                             vmin=vmin, vmax=vmax,
                             cmap=cmap, levels=levels)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Percentiles')
    log_image_to_wandb(fig, f"max_samples", imdir)


def figure_five(fake_u, train_u, imdir:str, channel=0,
                 cmap="Spectral_r", levels=20):
    """Plot the 32 most extreme train and generated percentiles."""
    from hazGAN import DiffAugment  

    policy = 'color,translation,cutout'

    # prep data to plot
    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    fake = DiffAugment(fake_u, policy).numpy()[..., channel]
    real = DiffAugment(train_u, policy).numpy()[..., channel]

    if fake.shape[0] < 32:
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plot specs
    fig = plt.figure(figsize=(16, 16), layout="tight")
    subfigs = fig.subfigures(2, 1, hspace=0.2)

    for subfig, item in zip(subfigs, samples.items()):
        axs = subfig.subplots(4, 8, sharex=True, sharey=True,
                                    gridspec_kw={'hspace': 0, 'wspace': 0})
        label = item[0]
        sample = item[1]
        vmin = sample.min()
        vmax = sample.max()
        for i, ax in enumerate(axs.flat):
            im = ax.contourf(lon, lat, sample[i, ...],
                             vmin=vmin, vmax=vmax,
                             cmap=cmap, levels=levels)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Augmented Percentiles')
    log_image_to_wandb(fig, f"augmented_percentiles", imdir)


## ----Old stuff (decide if needed later)---- ##
def add_watermark(ax, text):
        ax.text(-1, 0.01, text,
        fontsize=8, color='k', alpha=0.5,
        ha='right', va='bottom',
        transform=plt.gca().transAxes)


def add_colorbar(fig, im, ax, pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    fig.colorbar(im, cax=cax, orientation='vertical')


def discrete_colormap(data, nintervals, min=None, cmap="cividis", under='lightgrey'):
    cmap = getattr(mpl.cm, cmap)
    if min is not None:
        data = data[data >= min]
    bounds = np.quantile(data.ravel(), np.linspace(min, 1, nintervals))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under(under)
    return cmap, norm


def plot_generated_marginals(fake_data, start=0, nchannels=None, runname="", vmin=0, vmax=1):
    print(f"Range: [{fake_data.min():.2f}, {fake_data.max():.2f}]")

    if nchannels is None:
        nchannels = fake_data.shape[-1]

    ncols = 10
    fig, axs = plt.subplots(nchannels, ncols, layout='tight', figsize=(10, 3 * nchannels))

    if nchannels == 1:
        axs = axs[np.newaxis, :]
    for i in range(nchannels):
        for j in range(ncols):
            ax = axs[i, j]
            im = ax.imshow(fake_data[start + j, ..., i], cmap='Spectral_r', vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylabel(f'channel {i}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Quantile')
        plt.suptitle(f'Generated marginals {runname}')

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


# def compare_tails_plot(train_marginals, test_marginals, fake_marginals, x, y, params=None,
#                      thresh=None, channel=0, cmap='cividis', inverse_transform=True, evt_type="pot",
#                      figsize=(10, 10), s=10):
#     if channel == 0:
#         corrs = {'low': (220, 373), 'medium': (294, 189), 'high': (332, 311)}
#     elif channel == 1:
#         corrs = {'low': (121, 373), 'medium': (294, 189), 'high': (232, 276)}
#     elif channel == 2:
#         corrs = {'low': (121, 373), 'medium': (294, 189), 'high': (332, 311)}

#     fig, axs = plt.subplots(3, 3, figsize=figsize, layout='tight')

#     if inverse_transform:
#         train_quantiles = POT.inv_probability_integral_transform(train_marginals, x, y, params, evt_type, thresh)
#         test_quantiles = POT.inv_probability_integral_transform(test_marginals, x, y, params, evt_type, thresh)
#         fake_quantiles = POT.inv_probability_integral_transform(fake_marginals, x, y, params, evt_type, thresh)
#     else:
#         train_quantiles = train_marginals
#         test_quantiles = test_marginals
#         fake_quantiles = fake_marginals
#         if thresh is not None:
#             thresh = interpolate_thresholds(thresh, x, y)

#     if thresh is not None:
#         h, w, c = thresh.shape
#         thresh = thresh.reshape(h * w, c)

#     for i, sample_pixels in enumerate([*corrs.values()]):
#         ax = axs[i, :]
#         plot_sample_density(train_quantiles[..., channel], ax[0], sample_pixels=sample_pixels, cmap=cmap, s=s)
#         plot_sample_density(test_quantiles[..., channel], ax[1], sample_pixels=sample_pixels, cmap=cmap, s=s)
#         plot_sample_density(fake_quantiles[..., channel], ax[2], sample_pixels=sample_pixels, cmap=cmap, s=s)

#         ec = [*get_extremal_coeffs(train_marginals[..., channel], sample_pixels).values()][0]
#         ax[0].set_title(f'$\\chi_{{{sample_pixels[0], sample_pixels[1]}}}$: {2 - ec:.4f}')
#         ec = [*get_extremal_coeffs(test_marginals[..., channel], sample_pixels).values()][0]
#         ax[1].set_title(f'$\\chi_{{{sample_pixels[0], sample_pixels[1]}}}$: {2 - ec:.4f}')
#         ec = [*get_extremal_coeffs(fake_marginals[..., channel], sample_pixels).values()][0]
#         ax[2].set_title(f'$\\chi_{{{sample_pixels[0], sample_pixels[1]}}}$: {2 - ec:.4f}')
#         ax[0].set_ylabel(f'{channel_labels[channel]}')

#         if thresh is not None:
#             u = (thresh[sample_pixels[0], channel], thresh[sample_pixels[1], channel])

#             for a in ax:
#                 a.axhline(u[0], linestyle='dashed', color='k', label='Threshold')
#                 a.axvline(u[1], linestyle='dashed', color='k')

#         xmin = min(ax[0].get_xlim()[0], ax[1].get_xlim()[0], ax[2].get_xlim()[0])
#         xmax = max(ax[0].get_xlim()[1], ax[1].get_xlim()[1], ax[2].get_xlim()[1])
#         ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0])
#         ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1])
#         xlim = (xmin, xmax)
#         ylim = (ymin, ymax)

#         for a in ax:
#             a.set_xlim(xlim)
#             a.set_ylim(ylim)
    
#     axs[2, 0].set_xlabel(f'{channel_labels[channel]}\nTrain set')
#     axs[2, 1].set_xlabel(f'{channel_labels[channel]}\nTest set')
#     axs[2, 2].set_xlabel(f'{channel_labels[channel]}\nGenerated set')

#     axs[0, 0].legend(loc='upper left')

#     for ax in axs.ravel():
#         ax.label_outer()
            
#     suptitle = f'Correlations: {channel_labels[channel]}'
#     suptitle = suptitle if inverse_transform else f'{suptitle} (uniform marginals)'
#     fig.suptitle(suptitle)
#     return fig


def plot_sample_density(data, ax, sample_pixels=None, cmap='cividis', s=10):
    """For generated quantiles."""
    h, w = data.shape[1:3]
    n = h * w

    if sample_pixels is None:
        sample_pixels_x = random.sample(range(n), 1)
        sample_pixels_y = random.sample(range(n), 1)
    else:
        assert sample_pixels[0] != sample_pixels[1]
        sample_pixels_x = [sample_pixels[0]]
        sample_pixels_y = [sample_pixels[1]]

    data_ravel = tf.reshape(data, [len(data), n])

    sample_x = tf.gather(data_ravel, sample_pixels_x, axis=1)
    sample_y = tf.gather(data_ravel, sample_pixels_y, axis=1)

    axtitle = f"Pixels ({sample_pixels_x[0]}, {sample_pixels_y[0]})"
    scatter_density(sample_x.numpy(), sample_y.numpy(), ax, title=axtitle, cmap=cmap, s=s)


def scatter_density(x, y, ax, title='', cmap='cividis', s=10):
    xy = np.hstack([x, y]).transpose()
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=s, cmap=cmap)
    ax.set_title(title)
    return ax


def scatter_density2(x, y, ax, title='', cmap='cividis'):
    """Sometimes first doesn't work -- need to resolve why later."""
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=10, cmap=cmap)
    ax.set_title(title)
    return ax


def compare_channels_plot(train_images, test_images, fake_data, cmap='cividis'):
    fig, axs = plt.subplots(3, 3, figsize=(15, 3))

    for i, j in enumerate([300, 201, 102]):

        n, h, w, c = train_images.shape
        data_ravel = tf.reshape(train_images, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 0], cmap=cmap)

        n, h, w, c = test_images.shape
        data_ravel = tf.reshape(test_images, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 1], cmap=cmap)

        n, h, w, c = fake_data.shape
        data_ravel = tf.reshape(fake_data, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 2], cmap=cmap)

        for ax in axs.ravel():
            ax.set_xlabel('u10')
            ax.set_ylabel('v10')
    return fig


def plot_one_hundred_images(fake_data, channel=0, suptitle="Generated marginals", **plot_kwargs):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    heights = [1] * 10 + [0.1]
    gs = gridspec.GridSpec(11, 10, wspace=0.1, hspace=0.1, height_ratios=heights)

    for i in range(100):
        ax = plt.subplot(gs[i])
        im = ax.imshow(fake_data[i, ..., channel], **plot_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_subplot(gs[-1, :])
    fig.colorbar(im, cax=cax, shrink=0.8, aspect=0.01, orientation="horizontal", label='Quantile')
    fig.suptitle(suptitle)
    return fig


