"""
Make main paper figures.

>>> micromamba activate hazGAN
>>> python -m figures.py --domain gaussian
>>> python -m figures.py --domain rescaled
>>> python -m figures.py --domain uniform
>>> python -m figures.py --domain gumbel
"""
# %%
import os
import sys
from environs import Env
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from importlib import reload
from pathlib import Path
from argparse import ArgumentParser
from utils.analysis import load_samples

from hazGAN.plotting import fields
from hazGAN.plotting import spatial
from hazGAN.plotting import samples
from hazGAN.plotting import misc
from hazGAN.plotting import scatter


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'figure.titlesize': 8,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'normal',
    'legend.fontsize': 6,
    'font.family': 'sans-serif'
})


# settings for interactive script
global DOMAIN
DOMAIN = "gaussian"

# field metadata
savefigs = True
units_orig = ["m/s", "m", "Pa"]
units = ["m/s", "mm", "hPa"]
scales = [1, 1000, 0.01]
field_names = ["u10", "tp", "mslp"]
field_labels = ["wind speed", "precipitation", "pressure"]
field_cmaps = ["viridis", "PuBu", "OrRd_r"]
savefig_kws = dict(dpi=300, bbox_inches='tight', transparent=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--thresh", type=float, default=15., help="Max wind threshold (m/s)")
    parser.add_argument("-m", "--month", type=int, default=9, help="Month to plot (1-12)")
    parser.add_argument("-n", "--nyrs", type=int, default=500, help="Number of years of samples")
    parser.add_argument("-s", "--scaling", type=str, default="rp10000", choices=["rp10000", "minmax"])
    parser.add_argument("-d", "--domain", type=str, default=DOMAIN, choices=["rescaled", "uniform", "gaussian", "gumbel"])

    interactive = 'ipykernel' in sys.modules or hasattr(sys, 'ps1')
    
    if interactive:
        return parser.parse_args(args=[]) 
    else:
        return parser.parse_args()


def savefig(fig, outpath:str, savefigs:bool, savefig_kws:dict):
    """simple wrapper."""
    if savefigs:
        fig.savefig(outpath, **savefig_kws)
        print(f"saved figure to {outpath}")
    else:
        print("not saving figure. (savefigs is False)")


def export_nc(path, data, lats, lons, ny, nx, domain):
    print("warning: netcdf export domain is hardcoded to BoB.")
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
    ds.to_netcdf(path)
    print(f"Exported netCDF to {path}")
        


def save_stats_csv(path, stats_dict):
    with open(path, "w") as f:
        headers = ["metric"] + list(stats_dict.keys())
        f.write(",".join(headers) + "\n")
        metrics = list(next(iter(stats_dict.values())).keys())
        for metric in metrics:
            row = [metric]
            for section in stats_dict.keys():
                value = stats_dict[section][metric]
                val_str = f"{value:.4f}" if isinstance(value, (float, np.float64)) else str(value)
                row.append(val_str)
            f.write(",".join(row) + "\n")


if __name__ == "__main__":

    np.random.seed(42)

    env = Env()
    env.read_env()

    args = parse_args()
    del DOMAIN

    # configure paths   
    train_dir = Path(env.str("TRAINDIR"))
    samples_dir = Path(env.str("SAMPLES_DIR")) / args.scaling / args.domain / "npy"
    figdir = Path(env.str("FIG_DIR"))
    figdir.mkdir(parents=True, exist_ok=True)
    medians_path = train_dir / "data.nc"

    print(f"{str(train_dir)=}")
    print(f"{str(samples_dir)=}")
    print(f"{str(figdir)=}\n")

    # list sample files
    all_sample_files = os.listdir(samples_dir)
    print(f"Found {len(all_sample_files)} samples.\n")

    # loading takes a while
    data = load_samples(
        samples_dir, train_dir,
        threshold=args.thresh, nyrs=args.nyrs,
        domain=args.domain, scaling=args.scaling,
        make_benchmarks=False
    )

    # export netcdf if doesn't exist
    outnc = samples_dir.parent / "nc" / "data.nc"

    if not os.path.exists(outnc):
        outnc.parent.mkdir(parents=True, exist_ok=True)
        export_nc(outnc, data, lats=[5, 25], lons=[80, 95], ny=64, nx=64, domain=args.domain)
        print(f"saved netCDF to {outnc}")

    # unpack data
    u_trn = data['training']['u']
    y_trn = data['training']['y']
    x_trn = data['training']['x']
    cop_trn = data['training']['copula']
    msk_trn = data['training']['mask']

    u_gen = data['samples']['u']
    y_gen = data['samples']['y']
    x_gen = data['samples']['x']
    cop_gen = data['samples']['copula']
    msk_gen = data['samples']['mask']

    # load climatology data
    medians = xr.open_dataset(medians_path)["medians"]
    medians = medians.isel(month=args.month).values

    # %% ==============================================================
    # update results dict
    statspath = figdir / "stats" / f"{args.domain}.csv"
    statspath.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    
    results["climatology (median)"] = {}
    results["climatology (median)"]["wind speed"] = f"{np.quantile(medians[..., 0], 0.5):.2f} m/s"
    results["climatology (median)"]["precipitation"] = f"{np.quantile(medians[..., 1], 0.5):.2f} m"
    results["climatology (median)"]["pressure"] = f"{np.quantile(medians[..., 2], 0.5):.2f} Pa"

    results["median training anomaly"] = {}
    results["median training anomaly"]["wind speed"] = f"{np.quantile(x_trn[..., 0], 0.5):.2f} m/s"
    results["median training anomaly"]["precipitation"] = f"{np.quantile(x_trn[..., 1], 0.5):.2f} m"
    results["median training anomaly"]["pressure"] = f"{np.quantile(x_trn[..., 2], 0.5):.2f} Pa"

    # add medians back to anomalies
    x_trn = x_trn.copy() + medians
    x_gen = x_gen.copy() + medians

    # invert pressure
    x_trn[..., 2] *= -1
    x_gen[..., 2] *= -1

    # clip all at zero for cases medians don't handle
    x_trn = np.clip(x_trn, a_min=0, a_max=None)
    x_gen = np.clip(x_gen, a_min=0, a_max=None)

    results["median training"] = {}
    results["median training"]["wind speed"] = f"{np.quantile(x_trn[..., 0], 0.5):.2f} m/s"
    results["median training"]["precipitation"] = f"{np.quantile(x_trn[..., 1], 0.5):.2f} m"
    results["median training"]["pressure"] = f"{np.quantile(x_trn[..., 2], 0.5):.2f} Pa"

    save_stats_csv(statspath, results)
    print(f"Saved summary statistics to {statspath}")
    del results, statspath

    # %% ==============================================================
    # saffir-simpson barchart
    barcharts = False
    if barcharts:
        reload(misc)

        outpath = figdir / "saffir-simpson" / (args.domain + ".pdf")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        statspath = figdir / "saffir-simpson" / (args.domain + ".csv")

        results = {}

        fig, ax = misc.saffirsimpson_barchart(
            x_gen, x_trn, title="", scale="fives",
            barwidth=0.4, figsize=(4, 2), density=True,
            results=results # mutable dict
        )

        savefig(fig, outpath, savefigs, savefig_kws)
        save_stats_csv(statspath, results)
        print(f"Saved figure to {outpath}")
        print(f"Saved summary statistics to {statspath}")
        del results, statspath
    
    # %% ==============================================================
    # wind maxima vs return period plot
    windprofiles = False
    if windprofiles:
        λ_trn = len(x_trn) / 81
        fig, ax = plt.subplots(figsize=(2.5, 1.5), constrained_layout=True)
        transpose = False

        profile_kws = dict(ax=ax, windchannel=0, transpose=transpose)

        misc.windvreturnperiod(x_gen, λ_trn, label="HazGAN",
                               linewidth=2, linestyle='-', 
                               color='k',
                               **profile_kws)
        misc.windvreturnperiod(x_trn, λ_trn, label="ERA5",
                               color='gray', alpha=0.5,
                               marker='.', markersize=2,
                               markeredgecolor='gray',
                               linestyle='',
                               **profile_kws)   

        outpath = figdir / "wind-profile" / (args.domain + ".pdf")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        savefig(fig, outpath, savefigs, savefig_kws)
    
    # %% ==============================================================
    # Boulagiem (2022)-style scatterplots
    scatterplots = False
    if scatterplots:

        from hazGAN.constants import OBSERVATION_POINTS
        from hazGAN import op2idx
        reload(scatter)

        def get_ext_coeff(uniform_array, pixels:list):
            from hazGAN.statistics import inverted_frechet
            n, h, w, k = uniform_array.shape
            uniform_array = uniform_array.reshape(n, h * w, k)
            uniform_array = np.take(uniform_array, pixels, axis=1)
            frechet = inverted_frechet(uniform_array)
            minima = np.sum(np.min(frechet, axis=1), axis=0).astype(float)
            ecs = np.divide(n, minima, out=np.zeros_like(minima), where=minima != 0, dtype=float)
            return ecs
        
        outdir = figdir / "scatter-ops" / args.domain
        outdir.mkdir(parents=True, exist_ok=True)
        statspath = outdir / "smith1990.csv"

        results = {}

        op_options = [
            ["chittagong", "dhaka"],
            ["buoy_23007", "buoy_23008"],
            ["buoy_23218", "buoy_23223"]
        ]

        for op_pair in op_options:

            ops = op2idx(OBSERVATION_POINTS, x_trn[0, ..., 0], extent=[80, 95, 10, 25])
            
            op_a, op_b = op_pair
            pixels = [ops[op_a], ops[op_b]] # correlated

            for i in range(3):
                x_scaled = x_gen.copy() * scales[i]
                y_scaled = x_trn.copy() * scales[i]
                unit = units[i]

                fig = scatter.plot(
                    x_scaled, y_scaled, field=i,
                    pixels=pixels, s=1,
                    cmap='viridis',
                    xlabel=op_a.replace('_', ' ').title(),
                    ylabel=op_b.replace('_', ' ').title(),
                    figsize=(3,1.5),
                    suptitle=f"{field_labels[i].capitalize()} ({unit})"
                )
                
                outpath = outdir / f"{op_a}-{op_b}" / (field_names[i] + ".pdf") 
                outpath.parent.mkdir(parents=True, exist_ok=True)
                savefig(fig, outpath, savefigs, savefig_kws)
            
            ecs_trn = get_ext_coeff(u_trn, pixels)
            ecs_gen = get_ext_coeff(u_gen, pixels)

            results[f"{op_a}-{op_b}"] = {}

            results[f"{op_a}-{op_b}"]["u10_trn"] = f"{ecs_trn[0]:.4f}"
            results[f"{op_a}-{op_b}"]["u10_gen"] = f"{ecs_gen[0]:.4f}"
            results[f"{op_a}-{op_b}"]["tp_trn"] = f"{ecs_trn[1]:.4f}"
            results[f"{op_a}-{op_b}"]["tp_gen"] = f"{ecs_gen[1]:.4f}"
            results[f"{op_a}-{op_b}"]["mslp_trn"] = f"{ecs_trn[2]:.4f}"
            results[f"{op_a}-{op_b}"]["mslp_gen"] = f"{ecs_gen[2]:.4f}"
            
        save_stats_csv(statspath, results)
        print(f"Saved summary statistics to {statspath}")

        del results, statspath

    # %% ==============================================================
    # 64x64 plots of data
    sampleplots = False
    if sampleplots:
        reload(samples)

        def format_rp(value, tick_number):
            return f"{value:,.0f}"

        field = 0
        shuffle = False

        k = scales[field]
        metric = units[field]
        cmap = field_cmaps[field] 

        if shuffle:
            id_gen = np.random.permutation(u_gen.shape[0])
            id_trn = np.random.permutation(u_trn.shape[0])
        else:
            id_gen = np.arange(u_gen.shape[0])
            id_trn = np.arange(u_trn.shape[0])

        rp_trn = 1 / (1 - u_trn)
        rp_gen = 1 / (1 - u_gen)

        fig_y = samples.plot(y_gen[id_gen], y_trn[id_trn], field=field, title="", cmap=cmap)
        fig_u = samples.plot(u_gen[id_gen], u_trn[id_trn], field=field, title="", cbar_label="", cmap=cmap)
        fig_rp = samples.plot(rp_gen[id_gen], rp_trn[id_trn], field=field, title="", cbar_label="return period (years)", cmap=cmap, cbar_formatter=FuncFormatter(format_rp))
        fig_x = samples.plot(k*x_gen[id_gen], k*x_trn[id_trn], field=field, title="", cbar_label=metric, cmap=cmap, vmin=0)
    
        outdir = figdir / "samples-64x64" / args.domain
        outdir.mkdir(parents=True, exist_ok=True)

        outpath = os.path.join(outdir, f"{field_names[field]}_y.png")
        savefig(fig_y, outpath, savefigs, savefig_kws)
        outpath = os.path.join(outdir, f"{field_names[field]}_u.png")
        savefig(fig_u, outpath, savefigs, savefig_kws)
        outpath = os.path.join(outdir, f"{field_names[field]}_rp.png")
        savefig(fig_rp, outpath, savefigs, savefig_kws)
        outpath = os.path.join(outdir, f"{field_names[field]}_x.png")
        savefig(fig_x, outpath, savefigs, savefig_kws)
    
    # %% ==============================================================
    # wasserstein distance maps
    wassdists = False
    if wassdists:
        reload(fields)
        import cartopy.crs as ccrs


        outdir = figdir / "wasserstein"
        outdir.mkdir(parents=True, exist_ok=True)
        statspath = outdir / f"{args.domain}.csv"


        def _wasserstein_1d(x0, x1):# Sort both distributions
            x0_sorted = np.sort(x0)
            x1_sorted = np.sort(x1)

            num_quantiles = max(len(x0), len(x1))
            quantiles = np.linspace(0, 1, num_quantiles)
            
            x0_quantile_vals = np.quantile(x0_sorted, quantiles)
            x1_quantile_vals = np.quantile(x1_sorted, quantiles)
            
            return np.mean(np.abs(x0_quantile_vals - x1_quantile_vals))


        def wasserstein(x0, x1, normalize=True):
            _, h, w = x0.shape
            
            x0 = x0.reshape(-1, h * w)
            x1 = x1.reshape(-1, h * w)

            wassdists = []
            for i in range(h * w):
                x0_i = x0[:, i]
                x1_i = x1[:, i]
                
                wd = _wasserstein_1d(x0_i, x1_i)
                
                if normalize:
                    x0_std = np.std(x0_i)
                    wd = wd / x0_std if x0_std > 1e-9 else wd
                    
                wassdists.append(wd)
            wassdists = np.stack(wassdists, axis=0).reshape(h, w)
            return np.array(wassdists).reshape(h, w)

    
        wdists = []
        results = {}
        results["min"] = {}
        results["max"] = {}
        results["mean"] = {}
        for field in range(3):
            wdist = wasserstein(x_gen[..., field], x_trn[..., field], normalize=True)
            wdists.append(wdist)
            results["min"][field_names[field]] = f"{wdist.min():.4f}"
            results["max"][field_names[field]] = f"{wdist.max():.4f}"
            results["mean"][field_names[field]] = f"{wdist.mean():.4f}"
        
        vmin = min([wd.min() for wd in wdists])
        vmax = max([wd.max() for wd in wdists])
        
        fig, axs = plt.subplots(1, 3, figsize=(4, 1.5), constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()})
        for field in range(3):

            im = fields.contourmap(
                wdists[field], ax=axs[field], extent=[80, 95, 10, 25],
                cmap="viridis", vmin=vmin, vmax=vmax, levels=12
            )
            axs[field].set_title(field_labels[field].capitalize())

        cbar = fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.6, fraction=0.05, pad=0.04)
        cbar.set_label("W. dist.\n(x / std dev.)")
        
        fig.savefig(os.path.join(outdir, f"{args.domain}.png"), dpi=300, bbox_inches='tight', transparent=True)

        save_stats_csv(statspath, results)
        print(f"Saved summary statistics to {statspath}")
        del results, statspath
    
    # %% ==============================================================
    # inter-field correlation plots
    fieldcorrplots = True
    if fieldcorrplots:
        from itertools import combinations
        reload(fields)

        outdir = figdir / "corr-fields" / args.domain
        outdir.mkdir(parents=True, exist_ok=True)
        statspath = outdir / "summary.csv"

        results = {}

        for pair in combinations([0, 1, 2], 2):
            pair_str = f"{pair[0]}{pair[1]}"

            # Pearson plots
            results[f"pearson{pair_str}"] = {}
            fig_r, metrics_r = fields.plot(
                u_gen, u_trn, fields.pearson,
                fields=pair, figsize=.3,
                title="", cbar_label="r",
                cmap="viridis", vmin=-1, vmax=1
            )
            results[f"pearson{pair_str}"]["mae"] = metrics_r['mae']
            results[f"pearson{pair_str}"]["pearson"] = metrics_r['pearson']

            outpath = os.path.join(outdir, f"pearson_{pair_str}.png")
            savefig(fig_r, outpath, savefigs, savefig_kws)

            # Extremal dependence plots
            results[f"extremal{pair_str}"] = {}
            fig_χ, metrics_χ = fields.plot(
                u_gen, u_trn,
                fields.extcorrboot,
                # fields.extcorr, # compare results; if similar, don't bootstrap fields
                fields=pair, figsize=.3,
                title="", cbar_label="χ(u)",
                vmin=0, vmax=1,
                cmap="viridis",
            )
            results[f"extremal{pair_str}"]["mae"] = metrics_χ['mae']
            results[f"extremal{pair_str}"]["pearson"] = metrics_χ['pearson']

            outpath = os.path.join(outdir, f"extcorr_{pair_str}.png")
            savefig(fig_χ, outpath, savefigs, savefig_kws)


        save_stats_csv(statspath, results)
        print(f"saved summary statistics to {statspath}")

    # %% ==============================================================
    # spatial correlation plots
    spatialcorrplots = True
    extcorrplot = True

    # NOTE: this section is computationally demanding
    # using two things to speed it up:
    # 1) use low-res data (every second pixel)
    # 2) use numba-optimized extremal correlation function

    if spatialcorrplots:
        reload(spatial)

        outdir = figdir / "corr-spatial" / args.domain
        outdir.mkdir(parents=True, exist_ok=True)
        statspath = outdir / "summary.csv"

        results = {}

        gen_lores = u_gen[:, ::2, ::2, :].copy()
        trn_lores = u_trn[:, ::2, ::2, :].copy()

        for k in [0, 1, 2]:

            # Pearson plots
            results[f"pearson{k}"] = {}
            fig_r, metrics_r = spatial.plot(
                gen_lores, trn_lores,
                spatial.pearson, field=k,
                figsize=.3, title="", cbar_label="r",
                cmap="viridis", vmin=-1, vmax=1
            )

            results[f"pearson{k}"]["mae"] = metrics_r['mae']
            results[f"pearson{k}"]["pearson"] = metrics_r['pearson']
            outpath = os.path.join(outdir, f"pearson_{k}.png")
            savefig(fig_r, outpath, savefigs, savefig_kws)

            if extcorrplot:
                results[f"extremal{k}"] = {}
                fig_χ, metrics_χ = spatial.plot(
                    gen_lores, trn_lores,
                    spatial.extcorrboot,
                    # spatial.extcorr, # bootstrap for final version; compare results first
                    field=k,
                    figsize=.3, title="", cbar_label="χ(u)",
                    cmap="viridis", vmin=0, vmax=1)
                
                results[f"extremal{k}"]["mae"] = metrics_χ['mae']
                results[f"extremal{k}"]["pearson"] = metrics_χ['pearson']

                outpath = os.path.join(outdir, f"extcorr_{k}.png")
                savefig(fig_χ, outpath, savefigs, savefig_kws)
                
        save_stats_csv(statspath, results)
        print(f"saved summary statistics to {statspath}")

# %% ==============================================================
 