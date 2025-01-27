import os
import random
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .constants import channel_labels
from .statistics import (
    invPIT,
    get_extremal_coeffs_nd,
    pairwise_extremal_coeffs,
    # interpolate_thresholds, # old, might not need
    # get_extremal_coeffs     # old, might not need
)

plt.rcParams.update({
    'font.family': 'Arial',  # or 'Helvetica'
    'font.size': 12
})

hist_kwargs = {"bins": 50, "color": "lightgrey", "edgecolor": "black", "alpha": 0.7}
save_kwargs = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0.1}


# plots for training script
def log_image_to_wandb(fig, name:str, dir:str, **kwargs):
    """Pass figure to wandb if available."""
    import wandb
    if wandb.run is not None:
        impath = os.path.join(dir, f"{name}.png")
        fig.savefig(impath, **kwargs)
        wandb.log({name: wandb.Image(impath)})
    else:
        print("Not logging figure, wandb not intialised.")


def cartopy_xlabel(ax, label):
    ax.text(0.5, 1.01, label, va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=8)
    

def cartopy_ylabel(ax, label):
    ax.text(-0.05, 0.5, label, va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=8)
    

def figure_one(fake_u:np.array, train_u:np.array, valid_u:np.array=None, imdir:str=None,
               id='', cmap='coolwarm_r') -> None:
    """Plot cross-channel extremal coefficients."""
    def get_channel_ext_coefs(x):
        _, h, w, _ = x.shape
        excoefs = get_extremal_coeffs_nd(x, [*range(h * w)])
        excoefs = np.array([*excoefs.values()]).reshape(h, w)
        return excoefs
    n = fake_u.shape[0]

    if valid_u is not None:
        ncolumns = 3
    else:
        ncolumns = 2
    
    excoefs_train = get_channel_ext_coefs(train_u)
    excoefs_fake = get_channel_ext_coefs(fake_u)

    cmap = getattr(plt.cm, cmap)
    vmin = min(
        np.nanmin(excoefs_train),
        np.nanmin(excoefs_fake)
        )
    vmax = max(
        np.nanmax(excoefs_train),
        np.nanmax(excoefs_fake)
    )
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig = plt.figure(figsize=(4 * ncolumns, 3.5))
    gs = fig.add_gridspec(1, ncolumns + 1, width_ratios=[1] * ncolumns + [0.05], wspace=0.02)

    plot_axes = [fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree()) for i in range(ncolumns)]
    cbar_ax = fig.add_subplot(gs[0, -1])
    ax = plot_axes + [cbar_ax]
    
    # plot coefficients
    extent = [80, 95, 10, 25]
    transform = ccrs.PlateCarree()

    # add contour lines
    levels = np.linspace(vmin, vmax, 20)
    im = ax[0].contourf(excoefs_train, levels=levels, extent=extent, transform=transform, cmap=cmap, extend='both')
    im = ax[-2].contourf(excoefs_fake, levels=levels, extent=extent, transform=transform, cmap=cmap, extend='both')
    ax[0].contour(excoefs_train, levels=[3.], extent=extent, transform=transform, colors='k', linewidths=2, linestyles='dotted')
    ax[-2].contour(excoefs_fake, levels=[3.], extent=extent, transform=transform, colors='k', linewidths=2, linestyles='dotted')

    if valid_u is not None:
        excoefs_valid = get_channel_ext_coefs(valid_u)
        im = ax[1].imshow(excoefs_valid, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, transform=transform)
        ax[1].contour(excoefs_valid, levels=3, colors='k', linewidths=.5,
                    linestyles='dashed', extent=extent, transform=transform)
        ax[1].set_title('Test', fontsize=16)

    for a in ax[:-1]:
        a.set_yticks([])
        a.set_xticks([])
        a.add_feature(cfeature.COASTLINE, linewidth=.5)
        a.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax[0].set_title('Train', fontsize=16)
    ax[-2].set_title('Model', fontsize=16);

    cbar = fig.colorbar(im, cax=ax[-1], extend='both', orientation='vertical', format='%.0f')
    cbar.set_label('Extremal coefficient', rotation=270, labelpad=15)

    fig.suptitle('Cross-channel extremal coefficients', fontsize=18, y=1.05)
    
    if imdir is not None:
        log_image_to_wandb(fig, f"extremal_dependence{id}", imdir)
    
    return excoefs_train, excoefs_fake


def figure_two(fake_u:np.array, train_u:np.array, valid_u:np.array=None, imdir:str=None,
               channel=0, id={}, cmap='coolwarm_r', yflip=True) -> None:
    """Plot spatial extremal coefficients."""
    ecs_train = pairwise_extremal_coeffs(train_u.astype(np.float32)[..., channel])
    ecs_fake  = pairwise_extremal_coeffs(fake_u.astype(np.float32)[..., channel])
    
    if valid_u is not None:
        ncolumns = 3
        ecs_valid = pairwise_extremal_coeffs(valid_u.astype(np.float32)[..., channel])
    else:
        ncolumns = 2

    if yflip:
        ecs_train = np.flip(ecs_train, axis=0)
        ecs_fake = np.flip(ecs_fake, axis=0)
        if valid_u is not None:
            ecs_valid = np.flip(ecs_valid, axis=1)
    
    cmap = getattr(plt.cm, cmap)
    vmin = min(
        np.nanmin(ecs_train),
        np.nanmin(ecs_fake)
    )
    vmax = max(
        np.nanmax(ecs_train),
        np.nanmax(ecs_fake)
    )

    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))
    
    fig, axs = plt.subplots(1, ncolumns + 1, figsize=(4 * ncolumns, 3.5),
                        gridspec_kw={
                            'wspace': .02,
                            'width_ratios': [1] * ncolumns +[.05]}
                            )

    im = axs[0].imshow(ecs_train, cmap=cmap, vmin=1, vmax=2)
    im = axs[-2].imshow(ecs_fake, cmap=cmap, vmin=1, vmax=2)

    if valid_u is not None:
        im = axs[1].imshow(ecs_valid, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].set_title("Test", fontsize=16)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_xaxis()
    
    cbar = fig.colorbar(im, cax=axs[-1], extend='both', orientation='vertical')
    cbar.set_label('Extremal coefficient', rotation=270, labelpad=15)

    axs[0].set_title("Train", fontsize=16)
    axs[-2].set_title("Model", fontsize=16)
    axs[0].set_ylabel(channel_labels[channel].capitalize(), fontsize=18);

    if imdir is not None:
        log_image_to_wandb(fig, f"spatial_dependence{id}", imdir)


def figure_three(fake_u:np.array, train_u:np.array, imdir:str=None, channel=0,
                 cmap="Spectral_r", levels=20, contour=True, id='') -> None:
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

    # set up plotting function
    if contour:
        plot = lambda ax: partial(ax.contourf, X=lon, Y=lat, levels=levels, cmap=cmap)
    else:
        plot = lambda ax: partial(ax.imshow, cmap=cmap)

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
            im = plot(ax)(sample[i, ...], vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()

        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Percentiles')

    if imdir is not None:
        log_image_to_wandb(fig, f"max_percentiles{id}", imdir)
    return fig


def figure_four(fake_u, train_u, train_x, params, imdir:str=None,
                channel=0, cmap="Spectral_r", levels=20,
                contour=True, id='') -> None:
    """Plot the 32 most extreme train and generated anomalies."""
    # prep data to plot
    fake_maxima = np.max(fake_u[..., channel], axis=(1, 2))
    real_maxima = np.max(train_u[..., channel], axis=(1, 2))

    fake = invPIT(fake_u, train_x, params)
    real = invPIT(train_u, train_x, params)
    fake = fake[..., channel]
    real = train_x[..., channel]

    lon = np.linspace(80, 95, 22)
    lat = np.linspace(10, 25, 18)
    lon, lat = np.meshgrid(lon, lat)
    
    if fake.shape[0] < 32:
        print("Not enough fake samples for figure four, duplicating.")
        fake = np.tile(
            fake,
            reps=(int(np.ceil(32 / fake.shape[0])), 1, 1)
            )
    if real.shape[0] < 32:
        print("Not enough real samples for figure four, duplicating.")
        real = np.tile(
            real,
            reps=(int(np.ceil(32 / real.shape[0])), 1, 1)
            )

    fake_sorting = np.argsort(fake_maxima)[::-1]
    fake = fake[fake_sorting, ...]

    real_sorting = np.argsort(real_maxima)[::-1]
    real = real[real_sorting, ...]

    samples = {'Generated samples': fake, "Training samples": real}

    # set up plotting function
    if contour:
        plot = lambda ax: partial(ax.contourf, X=lon, Y=lat, levels=levels, cmap=cmap)
    else:
        plot = lambda ax: partial(ax.imshow, cmap=cmap)

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
            im = plot(ax)(sample[i, ...], vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_yaxis()
        
        subfig.suptitle(label, y=1.04, fontsize=24)
        subfig.subplots_adjust(right=.99)
        cbar_ax = subfig.add_axes([1., .02, .02, .9]) 
        subfig.colorbar(im, cax=cbar_ax)

    fig.suptitle('Percentiles')

    if imdir is not None:
        log_image_to_wandb(fig, f"max_samples{id}", imdir)


def crossfield_correlations(x, dataset:str='', fields=None):
    x       = x.copy()
    fields  = fields or range(x.shape[-1])
    npixels = x.shape[1] * x.shape[1]
    nfields = x.shape[3]
    height  = x.shape[1]
    width   = x.shape[2]

    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

    def pixelcorr(array, axis):
        array = array[:, axis, fields].copy()
        corr  = np.corrcoef(array, rowvar=False)
        return corr 

    corrs = []
    for i in range(npixels):
        corr = pixelcorr(x, i)
        corrs.append(corr)

    corrs = np.stack(corrs, axis=0)
    corrs = corrs.reshape(height, width, nfields, nfields)
    
    extent = [80, 95, 10, 25]
    transform = ccrs.PlateCarree()

    fig, axs = plt.subplots(nfields, nfields, figsize=(5, 5),
                            sharex=True, sharey=True,
                            gridspec_kw={'hspace': .01, 'wspace': .01},
                            subplot_kw={'projection': ccrs.PlateCarree()}
                            )
    
    for i in range(nfields):
        for j in range(nfields):
            ax = axs[i, j]
            im = ax.contourf(corrs[..., i, j],
                            extent=extent, transform=transform,
                            cmap="Spectral_r", levels=np.linspace(0, 1, 20),
                            )
            
            ax.label_outer()
            ax.add_feature(cfeature.COASTLINE, linewidth=.5)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        
            if j == 0:
                cartopy_ylabel(ax, channel_labels[i])
            if i == 0:
                cartopy_xlabel(ax, channel_labels[j])

    cbar = fig.colorbar(im, ax=list(axs.flat), orientation='horizontal', format='%.2f', fraction=.05, pad=.05)
    cbar.set_label('Correlation')
    fig.suptitle(f"{dataset.capitalize()} cross-field correlations", fontsize=18)


def spatial_correlations(generated, training):
    """Plot spatial correlations for each field."""
    def one_plot(array, field, ax=None):
        """Plot spatial correlations for a single field."""
        array = array[..., field].copy()
        n, h, w = array.shape
        array = array.reshape(n, h * w)
        corrs = np.corrcoef(array.T)
        cmap = plt.get_cmap("Spectral_r")
        cmap.set_over(cmap(1.))
        cmap.set_under(cmap(0.))
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(corrs, vmin=0., vmax=.1, cmap="Spectral_r")
        return im 
        
    nfields = training.shape[-1]
    fig, axs = plt.subplots(2, 3, figsize=(6, 4.5),
                            gridspec_kw={'hspace': 0., 'wspace': 0.})
    for i in range(nfields):
        im = one_plot(training, i, ax=axs[0, i])
        im = one_plot(generated, i, ax=axs[1, i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
    
    axs[0, 0].set_ylabel("Training", fontsize=12)
    axs[1, 0].set_ylabel("Generated", fontsize=12)
    axs[0, 0].set_title("Wind speed", fontsize=12)
    axs[0, 1].set_title("Precipitation", fontsize=12)
    axs[0, 2].set_title("MSLP", fontsize=12)

    # add left/under extend to colorbar
    cbar = fig.colorbar(im, ax=list(axs.flat), orientation='horizontal',
                        extend='min', aspect=30, fraction=.03, pad=0.05);
    cbar.set_label("Pearson correlation", fontsize=12);
    fig.suptitle("Spatial correlations", fontsize=13);


## - - - - Older stuff (decide if needed later) - - - - - - - - - - - - - - - - - - |
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



def plot_sample_density(data, ax, sample_pixels=None, cmap='cividis', s=10):
    """Scatterplot for two variables, coloured by density."""
    h, w = data.shape[1:3]
    n = h * w

    if sample_pixels is None:
        sample_pixels_x = random.sample(range(n), 1)
        sample_pixels_y = random.sample(range(n), 1)
    else:
        assert sample_pixels[0] != sample_pixels[1]
        sample_pixels_x = [sample_pixels[0]]
        sample_pixels_y = [sample_pixels[1]]

    data_ravel = np.reshape(data, [len(data), n])

    sample_x = np.take(data_ravel, sample_pixels_x, axis=1)
    sample_y = np.take(data_ravel, sample_pixels_y, axis=1)

    axtitle = f"Pixels ({sample_pixels_x[0]}, {sample_pixels_y[0]})"

    if not isinstance(sample_x, np.ndarray):
        sample_x = sample_x.numpy()
        sample_y = sample_y.numpy()

    scatter_density(sample_x, sample_y, ax, title=axtitle, cmap=cmap, s=s)


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
        data_ravel = np.reshape(train_images, [n, h * w, c])
        data_sample = np.take(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 0], cmap=cmap)

        n, h, w, c = test_images.shape
        data_ravel = np.reshape(test_images, [n, h * w, c])
        data_sample = np.take(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 1], cmap=cmap)

        n, h, w, c = fake_data.shape
        data_ravel = np.reshape(fake_data, [n, h * w, c])
        data_sample = np.take(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 2], cmap=cmap)

        for ax in axs.ravel():
            ax.set_xlabel('u10')
            ax.set_ylabel('v10')
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
#         train_quantiles = invPIT(train_marginals, x, y, params, evt_type, thresh)
#         test_quantiles = invPIT(test_marginals, x, y, params, evt_type, thresh)
#         fake_quantiles = invPIT(fake_marginals, x, y, params, evt_type, thresh)
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