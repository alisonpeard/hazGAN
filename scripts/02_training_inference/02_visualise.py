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

from utils.analysis import load_samples

from hazGAN.plotting import fields
from hazGAN.plotting import spatial
from hazGAN.plotting import samples
from hazGAN.plotting import misc
from hazGAN.plotting import scatter

FIELD     = 0
THRESHOLD = [None, 15.][1]
MONTH     = 9
NYEARS    = 500
DOMAIN    = ["uniform", "gaussian", "gumbel"][2]
SAMPLES   = f"/soge-home/projects/mistral/alison/hazGAN-data/stylegan_output/{DOMAIN}/gen"


savefigs = True
figdir = SAMPLES.replace("stylegan_output", "figures").replace("/gen", "")
os.makedirs(figdir, exist_ok=True)
savefig_kws = dict(dpi=300, bbox_inches='tight', transparent=True)


def savefig(fig, outpath:str, savefigs:bool, savefig_kws:dict):
    """simple wrapper."""
    if savefigs:
        fig.savefig(outpath, **savefig_kws)
        print(f"Saved figure to {outpath}")
    else:
        print("Not saving figure. (savefigs is False)")


def export_nc(data, lats, lons, ny, nx, domain):
    print("\nWARNING: export netCDF is hardcoded for BoB study.")
    lats = np.linspace(lats[0], lats[1], ny)
    lons = np.linspace(lons[0], lons[1], nx)
    samples_idx = np.arange(data['samples']['x'].shape[0])
    ds = xr.Dataset(
        {
            "data": (("sample", "lat", "lon", "field"), data['samples']['x']),
            "uniform": (("sample", "lat", "lon", "field"), data['samples']['u']),
        },
        coords={
            "lon": (("lon",), lons),
            "lat": (("lat",), lats),
            "field": (("field",), ["u10", "tp", "mslp"]),
            "sample": (("sample",), samples_idx),
        })
    ds.to_netcdf(f'/soge-home/projects/mistral/alison/hazGAN-data/{domain}.nc')
        

if __name__ == "__main__":

    env = Env()
    env.read_env()

    train_dir   = env.str("TRAINDIR")
    samples_dir = SAMPLES

    np.random.seed(42)

    # create data
    samples_dir = os.path.expanduser(samples_dir)
    train_dir   = os.path.expanduser(train_dir)


    # loading takes a while
    data = load_samples(
        samples_dir, train_dir,
        threshold=THRESHOLD, ny=NYEARS, domain=DOMAIN
    )
    
    u_trn = data['training']['u']
    y_trn = data['training']['y']
    x_trn = data['training']['x']
    mask_trn = data['training']['mask']

    u_gen = data['samples']['u']
    y_gen = data['samples']['y']
    x_gen = data['samples']['x']
    mask_gen = data['samples']['mask']

    #  add monthly medians to x data
    medians_path = os.path.join(train_dir, "data.nc")
    medians = xr.open_dataset(medians_path)["medians"]
    medians["month"] = medians["time.month"]
    month_mask = medians["month"] == MONTH
    medians = medians.isel(time=month_mask)
    medians = medians.mean(dim='time').values

    print("\nClimatology")
    print(f"wind speed:     {np.quantile(medians[..., 0], 0.5):.2f} m/s")
    print(f"precipitation:  {np.quantile(medians[..., 1], 0.5):.2f} m")
    print(f"pressure:      {np.quantile(medians[..., 2], 0.5):.2f} Pa")

    print("\nMedian training anomalies:")
    print(f"wind speed:     {np.quantile(x_trn[..., 0], 0.5):.2f} m/s")
    print(f"precipitation:  {np.quantile(x_trn[..., 1], 0.5):.2f} m")
    print(f"pressure:       {np.quantile(x_trn[..., 2], 0.5):.2f} Pa")

    x_trn += medians
    x_gen += medians

    # for pressure, invert sign
    x_trn[..., 2] *= -1
    x_gen[..., 2] *= -1

    print(f"\nTraining medians with climatology:")
    print(f"wind speed:     {np.quantile(x_trn[..., 0], 0.5):.2f} m/s")
    print(f"precipitation:  {np.quantile(x_trn[..., 1], 0.5):.2f} m")
    print(f"pressure:       {np.quantile(x_trn[..., 2], 0.5):0.2f} Pa")

    # quick histogram of field maxima
    FIELD = 0
    ymax_trn = np.max(y_trn[..., FIELD], axis=(1, 2))
    ymax_gen = np.max(y_gen[..., FIELD], axis=(1, 2))

    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(ymax_gen, bins=25, density=True, alpha=0.5, label='Generated')
    plt.hist(ymax_trn, bins=25, density=True, alpha=0.5, label='ERA5')
    plt.xlabel(f"maxima, field: {FIELD}")
    plt.ylabel("Density");

    # %% barchart
    if True:
        reload(misc)
        fig, ax = misc.saffirsimpson_barchart(x_gen, x_trn, title="", scale="fives")
        fig.tight_layout()
        outpath = os.path.join(figdir, "saffir-simpson.png")
        savefig(fig, outpath, savefigs, savefig_kws)
    
    # %% wind maxima vs return period plot
    if True:
        _lambda = len(x_trn) / 81
        fig, ax = plt.subplots(figsize=(4, 4))
        transpose = False

        profile_kws = dict(ax=ax, windchannel=0, transpose=transpose)

        misc.windvsreturnperiod(x_gen, _lambda,
                                label="Generated", linewidth=3, color=misc.yellows[2],
                                linestyle='-', **profile_kws)
        misc.windvsreturnperiod(x_trn, _lambda,
                                label="Training", color='k', marker='o', linestyle='',
                                markeredgecolor='white', markeredgewidth=0.25, alpha=.7,
                                markersize=5, linewidth=0.5, **profile_kws)   

        outpath = os.path.join(figdir, "wind_return_period.png")
        savefig(fig, outpath, savefigs, savefig_kws)
    
    # %% Brown-Resnick style scatterplots
    if True:
        from hazGAN.constants import OBSERVATION_POINTS
        from hazGAN import op2idx
        reload(scatter)

        op_names = [
            ["chittagong", "dhaka"],
            ["buoy_23007", "buoy_23008"]
            ][1]
        ops = op2idx(OBSERVATION_POINTS, x_trn[0, ..., 0], extent=[80, 95, 10, 25])
        
        op_a, op_b = op_names
        pixels = [ops[op_a], ops[op_b]] # correlated

        for i in range(3):
            field, label = [(0, "c"), (1, "b"), (2, "a")][i]

            fig = scatter.plot(x_gen, x_trn, field=field, pixels=pixels, s=10,
                        cmap='viridis', xlabel=op_a.replace('_', ' ').title(),
                        ylabel=op_b.replace('_', ' ').title())
            
            outpath = os.path.join(figdir, f"fig13{label}__{op_a}-{op_b}.png")
            savefig(fig, outpath, savefigs, savefig_kws)
        
        def get_ext_coeff(uniform_array, pixels:list):
            from hazGAN.statistics import inverted_frechet
            n, h, w, k = uniform_array.shape
            uniform_array = uniform_array.reshape(n, h * w, k)
            uniform_array = np.take(uniform_array, pixels, axis=1)
            frechet = inverted_frechet(uniform_array)
            minima = np.sum(np.min(frechet, axis=1), axis=0).astype(float)
            ecs = np.divide(n, minima, out=np.zeros_like(minima), where=minima != 0, dtype=float)
            return ecs

        ecs_trn = get_ext_coeff(u_trn, pixels)
        ecs_gen = get_ext_coeff(u_gen, pixels)

        print("\nExtremal dependence coefficients at observation points:")
        print(f"Fields: 0: wind speed, 1: precipitation, 2: pressure")
        print("training:", ecs_trn)
        print("generated:", ecs_gen)

    # %% Make 64x64 plots of data
    if True:
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
            CMAP   = cmo.diff_r

        if False:
            # shuffle both in case they were sorted earlier
            id_gen = np.random.permutation(u_gen.shape[0])
            id_trn = np.random.permutation(u_trn.shape[0])
        else:
            # or don't
            id_gen = np.arange(u_gen.shape[0])
            id_trn = np.arange(u_trn.shape[0])
        
        figa = samples.plot(y_gen[id_gen], y_trn[id_trn], field=FIELD, title="", cmap=CMAP, ndecimals=0)
        figb = samples.plot(u_gen[id_gen], u_trn[id_trn], field=FIELD, title="", cbar_label="", cmap=CMAP, ndecimals=1)
        figc = samples.plot(x_gen[id_gen], x_trn[id_trn], field=FIELD, title="", cbar_label=METRIC, cmap=CMAP, vmin=0)
    
        outpath = os.path.join(figdir, f"samples_{FIELD}_gumbel.png")
        savefig(figa, outpath, savefigs, savefig_kws)
        outpath = os.path.join(figdir, f"samples_{FIELD}_uniform.png")
        savefig(figb, outpath, savefigs, savefig_kws)
        outpath = os.path.join(figdir, f"samples_{FIELD}.png")
        savefig(figc, outpath, savefigs, savefig_kws)

    # %% - - - - - Plot inter-field correlations - - - - - - - - - - - - - - -
    FIELD_LABELS = ["wind speed", "precipitation", "sea-level pressure"]

    if True:
        reload(fields)

        FIELDS = [0, 1]

        print(f"\nTail dependence coefficient:")
        figa, metrics_a = fields.plot(u_gen, u_trn, fields.tail_dependence, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$\hat\lambda$",
                    cmap="Spectral", vmin=0, vmax=1)

        print(f"\nPearson:")
        figb, metrics_b = fields.plot(u_gen, u_trn, fields.pearson, fields=FIELDS, figsize=.6,
                    title="", cbar_label=r"$r$", cmap="Spectral_r", vmin=-1, vmax=1)

        outpath = os.path.join(figdir, "fig12a.png")
        savefig(figa, outpath, savefigs, savefig_kws)
        outpath = os.path.join(figdir, "fig12b.png")
        savefig(figb, outpath, savefigs, savefig_kws)

        # print text summary for report
        text = f"\nThe correlation between the ERA5 and GAN-generated correlation fields " \
        f"for {FIELD_LABELS[FIELDS[0]]} and {FIELD_LABELS[FIELDS[1]]} is {metrics_b['pearson']:.3f} " \
        f"(MAE = {metrics_b['mae']:.3f}). "

        text += f"\nFor the extremes, inter-variable correlation between the ERA5 and GAN-generated fields of tail dependence coefficients " \
                f"for {FIELD_LABELS[FIELDS[0]]} and {FIELD_LABELS[FIELDS[1]]} is {metrics_a['pearson']:.3f} " \
                f"(MAE = {metrics_a['mae']:.3f})."

        print(text)
    # %% - - - - - Plot spatial correlations - - - - - - - - - - - - - - - - -
    if True:
        reload(spatial)

        for FIELD in [0]: #, 1, 2]:
            figa, metrics_a = spatial.plot(u_gen , u_trn, spatial.pearson, field=FIELD, figsize=.6,
            title="", cbar_label=r"$r$", cmap="Spectral_r", vmin=-1, vmax=1)

            outpath = os.path.join(figdir, f"fig11a_{FIELD}.png")
            savefig(figa, outpath, savefigs, savefig_kws)

            # %%  extremal coefficients (takes a minute)
            figb, metrics_b = spatial.plot(u_gen, u_trn, spatial.tail_dependence, field=FIELD, figsize=.6,
            title="", cbar_label=r"$\hat\lambda$", cmap="Spectral", vmin=0, vmax=1)

            outpath = os.path.join(figdir, f"fig11b_{FIELD}.png")
            savefig(figb, outpath, savefigs, savefig_kws)

            text = f"\nThe spatial correlation between the ERA5 and GAN-generated fields " \
                f"for {FIELD_LABELS[FIELD]} is {metrics_a['pearson']:.3f} " \
                f"(MAE = {metrics_a['mae']:.3f})."
            
            text += f"\nFor the extremes, spatial correlation between the ERA5 and GAN-generated fields of tail dependence coefficients " \
                    f"for {FIELD_LABELS[FIELD]} is {metrics_b['pearson']:.3f} " \
                    f"(MAE = {metrics_b['mae']:.3f})."
            
            print(text)

    # %% 
 