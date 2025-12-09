"""Load the generated samples and training data and plot metrics.

00030: new data for 300 epochs
"""
# %%
import os
from environs import Env
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from importlib import reload

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
MODEL     = 30 # 24 used for Zenodo / NHESS submission, 2 looks okay (5000 samples)
MODEL     = str(MODEL).zfill(5) if isinstance(MODEL, int) else MODEL
THRESHOLD = 15. # None to model all storms
TYPE      = ["results", "trunc-1_0"][1]
MONTH     = 9 #"September"
NYEARS    = 500

#  begin script
if __name__ == "__main__":
    # set up environment
    env = Env()
    env.read_env()

    samples_dir = env.str("SAMPLES_DIR")
    train_dir    = env.str("TRAINDIR")

    # set seed for np.random
    np.random.seed(42)

    #  - - - - - - load training logs - - - - - - - - - - - - - - - - - - - - -
    try:
        metrics_path = os.path.join(samples_dir, MODEL, 'stats.jsonl')
        metrics, fig = analyze_stylegan_training(metrics_path)
    
    except Exception as e:
        print(e)

    # create data
    samples_dir = os.path.expanduser(samples_dir)
    train_dir   = os.path.expanduser(train_dir)

    data = load_samples(samples_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE, ny=NYEARS)

    # %% NEW: September 2025 save netCDF
    if True:
        lats = np.linspace(10, 25, 64)
        lons = np.linspace(80, 95, 64)
        samples_idx = np.arange(data['samples']['data'].shape[0])
        ds = xr.Dataset(
            {
                "data": (("sample", "lat", "lon", "field"), data['samples']['data']),
                "uniform": (("sample", "lat", "lon", "field"), data['samples']['uniform']),
            },
            coords={
                "lon": (("lon",), lons),
                "lat": (("lat",), lats),
                "field": (("field",), ["u10", "tp", "mslp"]),
                "sample": (("sample",), samples_idx),
            })
        ds.to_netcdf(os.path.join(samples_dir, MODEL, f"samples_{TYPE}_thresh{THRESHOLD}_n{NYEARS}.nc"))
        
    # %% loading takes a while, checkpoint here
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

    #  add monthly medians to x data
    medians = xr.open_dataset(os.path.join(train_dir, "data.nc"))["medians"]
    medians["month"] = medians["time.month"]
    month_time_mask = medians["month"] == MONTH
    medians = medians.isel(time=month_time_mask)
    medians = medians.mean(dim='time').values

    print(f"Median pressure {np.quantile(medians[..., 2], 0.5):.2f} Pa")
    print(f"Median wind speed {np.quantile(medians[..., 0], 0.5):.2f} m/s")
    print(f"Median precipitation {np.quantile(medians[..., 1], 0.5):.2f} m")

    print(f"Median pressure anomaly {np.quantile(x[..., 2], 0.5):.2f} Pa")
    print(f"Median wind speed anomaly {np.quantile(x[..., 0], 0.5):.2f} m/s")
    print(f"Median precipitation anomaly {np.quantile(x[..., 1], 0.5):.2f} m")

    # %%
    x += medians
    valid_x += medians
    samples_x += medians

    x[..., 2] *= -1
    valid_x[..., 2] *= -1
    samples_x[..., 2] *= -1

    # %% histograms
    if False:
        reload(misc)

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'

        FAKE, REAL = samples_x, x
        fig, ax = misc.saffirsimpson_barchart(FAKE, REAL, title="", scale="fives")
        fig.tight_layout()
        fig.savefig(os.path.join("..", "..", "..", "figures", "saffir-simpson.pdf"), dpi=300)


    # %% wind maxima vs return period plot
    if False:
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
    if False:
        from hazGAN.constants import OBSERVATION_POINTS
        from hazGAN import op2idx
        reload(scatter)

        op_names = [
            ["chittagong", "dhaka"],
            ["buoy_23007", "buoy_23008"]
            ][1]
        ops = op2idx(OBSERVATION_POINTS, x[0, ..., 0], extent=[80, 95, 10, 25])
        
        op_a, op_b = op_names
        pixels = [ops[op_a], ops[op_b]] # correlated
        # pixels = [op_names[0]], ops["buoy_23008"]] # correlated

        for i in range(3):
            field, label = [(0, "c"), (1, "b"), (2, "a")][i]

            fig = scatter.plot(samples_x, x, field=field, pixels=pixels, s=10,
                        cmap='viridis', xlabel=op_a.replace('_', ' ').title(),
                        ylabel=op_b.replace('_', ' ').title())
            
            fig.savefig(os.path.join("..", "..", "..", "figures", f"fig13{label}__{op_a}-{op_b}.png"), dpi=300, bbox_inches='tight')
        
        def get_ext_coeff(uniform_array, pixels:list):
            from hazGAN.statistics import inverted_frechet
            n, h, w, k = uniform_array.shape
            uniform_array = uniform_array.reshape(n, h * w, k)
            uniform_array = np.take(uniform_array, pixels, axis=1)
            frechet = inverted_frechet(uniform_array)
            minima = np.sum(np.min(frechet, axis=1), axis=0).astype(float)
            ecs = np.divide(n, minima, out=np.zeros_like(minima), where=minima != 0, dtype=float)
            return ecs

        ecs_era5 = get_ext_coeff(u, pixels)
        ecs_gan = get_ext_coeff(samples_u, pixels)

        print(f"Fields: 0: wind speed, 1: precipitation, 2: pressure")
        print("extremal coeffs ERA5:", ecs_era5)
        print("extremal coeffs samples:", ecs_gan)

    # %% Make 64x64 plots of data
    if False:
        import cmocean.cm as cmo
        reload(samples)

        FIELD = 0
        if FIELD == 0:
            METRIC = r'ms$^{-1}$'
            CMAP   = "viridis" # Spectral_r,  # "viridis", cmo.speed
        elif FIELD == 1:
            METRIC = 'm'
            CMAP   = cmo.rain
        elif FIELD == 2:
            METRIC = "Pa"
            CMAP   = cmo.diff

        if True:
            # shuffle both in case they were sorted earlier
            if False:
                id0 = np.random.permutation(samples_u.shape[0])
                id1 = np.random.permutation(u.shape[0])
        else:
            # or don't
            id0 = np.arange(samples_u.shape[0])
            id1 = np.arange(u.shape[0])
        
        if False:
            samples_x = np.clip(samples_x, 0, None)
            x = np.clip(x, 0, None)

        figa = samples.plot(samples_gumbel[id0], gumbel[id1], field=FIELD, title="", cmap=CMAP, ndecimals=0)
        figb = samples.plot(samples_u[id0], u[id1], field=FIELD, title="", cbar_label="", cmap=CMAP, ndecimals=1)
        figc = samples.plot(samples_x[id0], x[id1], field=FIELD, title="", cbar_label=METRIC, cmap=CMAP, vmin=0)
    
        # - - - - - - - Save figures - - - - - - - - - - - - - - - -
        figa.savefig(os.path.join("..", "..", "..", "figures", f"samples_{FIELD}_gumbel.png"), dpi=300)
        figb.savefig(os.path.join("..", "..", "..", "figures", f"samples_{FIELD}_uniform.png"), dpi=300)
        figc.savefig(os.path.join("..", "..", "..", "figures", f"samples_{FIELD}.png"), dpi=300, transparent=True, bbox_inches='tight')
    
    # %% - - - - - Plot inter-field correlations - - - - - - - - - - - - - - -
    if True:
        reload(fields)

        FIELDS = [0, 1]
        FIELD_LABELS = ["wind speed", "precipitation", "sea-level pressure"]

        print(f"\nTail dependence coefficient:")
        figa, metrics_a = fields.plot(samples_u, u, fields.tail_dependence, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$\hat\lambda$",
                    cmap="Spectral", vmin=0, vmax=1)

        print(f"\nPearson:")
        figb, metrics_b = fields.plot(samples_u, u, fields.pearson, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$r$", cmap="Spectral_r", vmin=-1, vmax=1)

        figa.savefig(os.path.join("..", "..", "..", "figures", "fig12a.png"), dpi=300, bbox_inches='tight')
        figb.savefig(os.path.join("..", "..", "..", "figures", "fig12b.png"), dpi=300, bbox_inches='tight')

        text = f"The correlation between the ERA5 and GAN-generated correlation fields " \
        f"\nfor {FIELD_LABELS[FIELDS[0]]} and {FIELD_LABELS[FIELDS[1]]} is {metrics_b['pearson']:.3f} " \
        f"\n(MAE = {metrics_b['mae']:.3f}). "

        text += f"\nFor the extremes, inter-variable correlation between the ERA5 and GAN-generated fields of tail dependence coefficients " \
                f"\nfor {FIELD_LABELS[FIELDS[0]]} and {FIELD_LABELS[FIELDS[1]]} is {metrics_a['pearson']:.3f} " \
                f"\n(MAE = {metrics_a['mae']:.3f})."

        print(text)
    # %% - - - - - Plot spatial correlations - - - - - - - - - - - - - - - - -
    if False:
        reload(spatial)

        for FIELD in [2, 1, 0]:
            figa, metrics_a = spatial.plot(samples_u, u, spatial.pearson, field=FIELD, figsize=.6,
            title="", cbar_label=r"$r$", cmap="Spectral_r", vmin=-1, vmax=1)

            figa.savefig(os.path.join("..", "..", "..", "figures", "fig11a.png"), dpi=300, bbox_inches='tight')

            # extremal coefficients (take a hot minute)
            figb, metrics_b = spatial.plot(samples_u, u, spatial.tail_dependence, field=FIELD, figsize=.6,
            title="", cbar_label=r"$\hat\lambda$", cmap="Spectral", vmin=0, vmax=1)

            figb.savefig(os.path.join("..", "..", "..", "figures", "fig11b.png"), dpi=300, bbox_inches='tight')

            text = f"The spatial correlation between the ERA5 and GAN-generated fields " \
                f"for {FIELD_LABELS[FIELD]} is {metrics_a['pearson']:.3f} " \
                f"(MAE = {metrics_a['mae']:.3f})."
            
            text += f"For the extremes, spatial correlation between the ERA5 and GAN-generated fields of tail dependence coefficients " \
                    f"for {FIELD_LABELS[FIELD]} is {metrics_b['pearson']:.3f} " \
                    f"(MAE = {metrics_b['mae']:.3f})."
            
            print(text)

    # %% 
 