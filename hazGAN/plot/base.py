# %%
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',  # or 'Helvetica'
    'font.size': 12
})

CMAP   = "Spectral_r"
LABELS = ["wind speed [m]", "total precipitation [m]", "mslp [Pa]"]


def heatmap(array, ax=None, extent=[80, 95, 10, 25], transform=ccrs.PlateCarree(),
            cmap=CMAP, vmin=None, vmax=None, title=False):
    """Plot a heatmap with the coastline."""
    h, w = array.shape
    array = np.flip(array, axis=0) # imshow is upside down
    ax = ax or plt.axes(projection=transform)
    im = ax.imshow(array, extent=extent, transform=transform, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.add_feature(cfeature.COASTLINE, edgecolor='k', linewidth=0.5)
    ax.set_extent(extent)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=16);
    return im

def contourmap(array, ax=None, extent=[80, 95, 10, 25], transform=ccrs.PlateCarree(),
            cmap=CMAP, vmin=None, vmax=None, levels=15, extend="both", title=False):
    """Plot a heatmap with the coastline."""
    h, w = array.shape
    ax = ax or plt.axes(projection=transform)

    vmin = vmin or array.min()
    vmax = vmax or array.max()

    levels = np.linspace(vmin, vmax, levels)
    im = ax.contourf(array, extent=extent, transform=transform, cmap=cmap, levels=levels,
                     extend=extend)
    ax.add_feature(cfeature.COASTLINE, edgecolor='k', linewidth=0.5)
    ax.set_extent(extent)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=16);
    return im


def makegrid(rows, cols, figsize:float=1,
             cbar_width=.05, cbar_pad=.1):    

    total_width = cols + cbar_width + cbar_pad
    height = rows
    scale = 8 / max(total_width, height)
    figsize = (figsize * total_width * scale, figsize * height * scale)
    
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(rows, cols + 1, 
                         width_ratios=[1]*cols + [cbar_width],
                         wspace=0,
                         hspace=0,
                         left=0.01, right=0.99 - cbar_pad,
                         top=0.99, bottom=0.01)
    
    axes = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            axes[i,j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            # axes[i, j].set_aspect('auto')
    
    cax = fig.add_subplot(gs[:, -1])
    
    if rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    return fig, axes, cax


def makegrid(rows, cols, figsize:float=1,
             cbar_width=.05, cbar_pad=.1):    

    total_width = cols + cbar_width + cbar_pad
    height = rows
    scale = 8 / max(total_width, height)
    figsize = (figsize * total_width * scale, figsize * height * scale)
    
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(rows, cols + 2, 
                         width_ratios=[1]*cols + [cbar_pad, cbar_width],
                         wspace=0,
                         hspace=0,
                         left=0.01, right=0.99,
                         top=0.99, bottom=0.01)
    
    axes = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            axes[i,j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            # axes[i, j].set_aspect('auto')
    
    cax = fig.add_subplot(gs[:, -1])
    
    if rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    return fig, axes, cax

# %%
if __name__ == "__main__":
    # test functions
    import xarray as xr

    data = xr.open_dataset("/Users/alison/Documents/DPhil/paper1.nosync/training/constant-fields/terrain.nc")
    
    land      = data.mask.values.squeeze()
    oro       = data.elevation.values.squeeze()

    fig, axs, cax = makegrid(2, 2, figsize=.5, cbar_pad=0.)
    heatmap(land, ax=axs[0, 0])
    heatmap(oro, ax=axs[0, 1])
    contourmap(land, ax=axs[1, 0])
    im = contourmap(oro, ax=axs[1, 1])
    fig.colorbar(im, cax=cax, extend='both')
    fig.suptitle("Untitled", y=1.05)

    # %%
    fig, axs, cax = makegrid(2, 3)
    for ax in list(axs.flat):
        im = heatmap(oro, ax=ax)
    fig.colorbar(im, cax=cax, extend="both")

    # %% 
    fig, axs, cax = makegrid(3, 3)
    for ax in list(axs.flat):
        im = heatmap(oro, ax=ax)
    fig.colorbar(im, cax=cax, extend="both")

    # %%
    fig, axs, cax = makegrid(8, 8, cbar_width=.1, cbar_pad=.1)
    for ax in list(axs.flat):
        im = heatmap(oro, ax=ax)
    fig.colorbar(im, cax=cax, extend="both")


# %%
