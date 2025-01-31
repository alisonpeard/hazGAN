


## - - - - Older stuff (decide if needed later) - - - - - - - - - - - - - - - - - - |



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