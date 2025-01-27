from . import fields
from . import spatial
from . import samples
from .base import log_image_to_wandb
from .base import CMAP


def figure_one(fake_u, train_u, valid_u=None, imdir:str=None,
               cmap=CMAP, id='') -> None:
    
    fig = fields.plot(fake_u, train_u, fields.smith1990,
                      title="Extremal coefficient", cbar_label="Χ",
                      cmap=cmap, figsize=.8)

    if imdir is not None:
        log_image_to_wandb(fig, f"extremal_dependence{id}", imdir)


def figure_two(fake_u, train_u, valid_u=None, imdir:str=None,
               field=0, cmap=CMAP, id='') -> None:
    """Plot spatial extremal coefficients."""

    fig = spatial.plot(fake_u, train_u, spatial.smith1990, field=field,
                       title="Extremal correlation", cbar_label="Χ")

    if imdir is not None:
        log_image_to_wandb(fig, f"spatial_dependence{id}", imdir)


def figure_three(fake_u, train_u, imdir:str=None, field=0,
                 cmap=CMAP, id='') -> None:
    """Plot the 32 most extreme train and generated percentiles."""
    fig = samples.plot(fake_u, train_u, field=field,
                       title="Percentiles", cmap=cmap)
    
    if imdir is not None:
        log_image_to_wandb(fig, f"percentiles{id}", imdir)


def figure_four(fake_u, train_u, train_x, params, imdir:str=None,
                field=0, cmap=CMAP, id='') -> None:
    """Plot the 32 most extreme train and generated anomalies."""
    fig = samples.plot(fake_u, train_u, field=field, transform=samples.anomaly,
                       title="Anomalies", cmap=cmap, cbar_label="PIT",
                       reference=train_x, params=params)
    if imdir is not None:
        log_image_to_wandb(fig, f"anomalies{id}", imdir)