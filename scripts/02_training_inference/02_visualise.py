"""Load the generated samples and training data and plot metrics."""
# %%
import os
from environs import Env
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from utils.metrics_vis import analyze_stylegan_training
from utils.analysis import load_samples

from hazGAN.plotting import fields
from hazGAN.plotting import spatial
from hazGAN.plotting import samples
from hazGAN.plotting import misc
from hazGAN.plotting import scatter


# make sure font is Helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'


FIELD     = 0
MODEL     = 24
MODEL     = str(MODEL).zfill(5) if isinstance(MODEL, int) else MODEL
THRESHOLD = 15. # None to model all storms
TYPE      = "trunc-1_0"
MONTH     = 9 #"September"

#  begin script
if __name__ == "__main__":
    # set up environment
    env = Env()
    env.read_env()

    samples_dir = env.str("SAMPLES_DIR")
    data_dir    = env.str("DATADIR")
    br_dir      = env.str("BROWNRESNICK_DIR")
    train_dir    = env.str("TRAINDIR")

    #  - - - - - - load training logs - - - - - - - - - - - - - - - - - - - - -
    try:
        metrics_path = os.path.join(samples_dir, MODEL, 'stats.jsonl')
        metrics, fig = analyze_stylegan_training(metrics_path)
    
    except Exception as e:
        print(e)

    # create data
    data = load_samples(samples_dir, data_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE)
    u               = data['training']['uniform']
    gumbel          = data['training']['gumbel']
    x               = data['training']['data']
    mask            = data['training']['mask']

    valid_u         = data['valid']['uniform']
    valid_gumbel    = data['valid']['gumbel']
    valid_x         = data['valid']['data']
    valid_mask      = data['valid']['mask']

    samples_u       = data['samples']['uniform']
    samples_gumbel  = data['samples']['gumbel']
    samples_x       = data['samples']['data']
    samples_mask    = data['samples']['mask']

    # %% add monthly medians to x data
    medians = xr.open_dataset(os.path.join(train_dir, "data.nc"))["medians"]
    medians["month"] = medians["time.month"]
    month_time_mask = medians["month"] == MONTH
    medians = medians.isel(time=month_time_mask)
    medians = medians.mean(dim='time').values

    x += medians
    valid_x += medians
    samples_x += medians

    # %% histograms
    if True:
        # set font to Helvetica
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

        misc.invPITmaxima(samples_u, u, samples_gumbel, gumbel)

        # Plots of all pixel values, wind maxima and return periods
        FAKE, REAL = samples_x, x

        # misc.histogram(FAKE, REAL, misc.ravel, title="Histogram of all pixel values", xlabel="Pixel value [mps]")
        bar_width = 0.25
        fig, ax = misc.saffirsimpson_barchart(FAKE, REAL, bar_width=bar_width, title="")
        

        fig.tight_layout()
        fig.savefig(os.path.join(data_dir, "figures", "storm_distn.pdf"), dpi=300)

        # %%
        if True:
            misc.histogram(FAKE, REAL, misc.maxwinds, title="Distribution of footprint wind maxima (log-density)", xlabel="Maxima [mps]", yscale='log')
            misc.histogram(FAKE, REAL, misc.returnperiod, title="Storm return periods", xlabel="Return period [years]", _lambda=_lambda, yscale='log');
            print("Real maxima", np.max(REAL[..., 0]))
            print("Fake maxima", np.max(FAKE[..., 0]))


    # %% wind maxima vs return period plot
    if True:
        _lambda = len(x) / 81
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        transpose = False

        misc.windvsreturnperiod(FAKE, _lambda, ax=ax, windchannel=0, transpose=transpose,
                                label="Generated", linewidth=3, color=misc.yellows[2],
                                linestyle='-')
        misc.windvsreturnperiod(REAL, _lambda, ax=ax, windchannel=0, transpose=transpose,
                                label="Training", color='k', marker='o', linestyle='',
                                markeredgecolor='white', markeredgewidth=0.25, alpha=.7,
                                markersize=5, linewidth=0.5)
        
    
    # %% Brown-Resnick style scatterplots
    if True:
        from hazGAN.constants import OBSERVATION_POINTS
        from hazGAN import op2idx
        import pandas as pd

        ops = op2idx(OBSERVATION_POINTS, x[0, ..., 0], extent=[80, 95, 10, 25])
        
        pixels = [ops['chittagong'], ops['dhaka']]

        scatter.plot(samples_x, x, field=FIELD, pixels=pixels, s=10,
                     cmap='viridis', xlabel="Chittagong", ylabel="Dhaka")
        
        fig.savefig(os.path.join(data_dir, "figures", "scatters.png"), dpi=300)

    # %% Make 64x64 plots of data
    if True:
        FIELD = 0
        if FIELD == 0:
            METRIC = r'ms$^{-1}$'
            CMAP   = "viridis" # Spectral_r
        elif FIELD == 1:
            METRIC = 'm'
            CMAP   = "PuBu"
        elif FIELD == 2:
            METRIC = "Pa"
            CMAP   = "YlOrBr"

        if False:
            # shuffle both in case they were sorted earlier
            id0 = np.random.permutation(samples_u.shape[0])
            id1 = np.random.permutation(u.shape[0])
        else:
            # or don't
            id0 = np.arange(samples_u.shape[0])
            id1 = np.arange(u.shape[0])
        
        figa = samples.plot(samples_gumbel[id0], gumbel[id1], field=FIELD, title="", cmap=CMAP, ndecimals=0)
        figb = samples.plot(samples_u[id0], u[id1], field=FIELD, title="", cbar_label="", cmap=CMAP, ndecimals=1)
        figc = samples.plot(samples_x[id0], x[id1], field=FIELD, title="", cbar_label=METRIC, cmap=CMAP, alpha=1e-6);
    
        # - - - - - - - Save figures to Desktop - - - - - - - - - - - - - - - -
        figa.savefig(os.path.join(data_dir, "figures", f"samples_{FIELD}_gumbel.png"), dpi=300)
        figb.savefig(os.path.join(data_dir, "figures", f"samples_{FIELD}_uniform.png"), dpi=300)
        figc.savefig(os.path.join(data_dir, "figures", f"samples_{FIELD}.png"), dpi=300)
    
    # %% - - - - - Plot inter-field correlations - - - - - - - - - - - - - - -
    if True:
        # in probability space
        FIELDS = [0, 1]

        figa = fields.plot(samples_u, u, fields.smith1990, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$\hat\theta$",
                    cmap="Spectral", vmin=1, vmax=4)

        figb = fields.plot(samples_u, u, fields.pearson, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$r$", vmin=-1, vmax=1, cmap="Spectral_r")

        figa.savefig(os.path.join(data_dir, "figures", f"corr_smith_{FIELDS[0]}-{FIELDS[1]}.png"), dpi=300)
        figb.savefig(os.path.join(data_dir, "figures", f"corr_pearson_{FIELDS[0]}-{FIELDS[1]}.png"), dpi=300)
    # %% - - - - - Plot spatial correlations - - - - - - - - - - - - - - - - -
    if True:
        FIELD = 0
        figa = spatial.plot(samples_u, u, spatial.pearson, field=FIELD, figsize=.6,
        title="", cbar_label=r"$r$", cmap="Spectral_r", vmin=-1, vmax=1)

        figa.savefig(os.path.join(data_dir, "figures", f"corr_pearson_{FIELD}.png"), dpi=300)

        # extremal coefficients (take a minute)
        figb = spatial.plot(samples_u, u, spatial.smith1990, field=FIELD, figsize=.6,
        title="", cbar_label=r"$\hat\theta$", cmap="Spectral", vmin=1, vmax=4)

        figb.savefig(os.path.join(data_dir, "figures", f"corr_smiths_{FIELD}.png"), dpi=300)

    # %% 
 