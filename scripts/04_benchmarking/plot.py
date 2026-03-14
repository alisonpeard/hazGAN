"""Benchmark hazGAN against Heffernan & Tawn (2004) model."""
# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from environs import Env
import numpy as np


plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 6,
    'font.family': 'sans-serif'
})


path = "extcorrs.csv"


def hist(x, ax, **kwargs):
    x_data = df[x]
    ax.hist(x_data, **kwargs)


def scatter_with_corr(df, x, y, ax, **kwargs):
    x_data, y_data = df[x], df[y]
    
    r_pearson = df.corr().loc[x, y]
    r_squared = r_pearson**2
    mae = np.mean(np.abs(x_data - y_data))
    
    ax.scatter(x_data, y_data, **kwargs)

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='r',
            linewidth=0.5, zorder=0)
    
    w = 3
    label = (
        f"{'R':<{w}} = {r_pearson:.3f}\n"
        f"{'R^2':<{w}} = {r_squared:.3f}\n"
        f"{'MAE':<{w}} = {mae:.3f}"
    )
    ax.text(0.7, 0.3, label, transform=ax.transAxes, 
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.)
            )


if __name__ == "__main__":

    env = Env()
    env.read_env()

    df = pd.read_csv(path).dropna()

    figdir = Path(env.str("FIGDIR")) / "ht2004"
    figdir.mkdir(exist_ok=True, parents=True)

    scatter_kws = dict(s=1, alpha=1., color='k', edgecolor="none")

    # plot spatial extremal correlations
    fig, axs = plt.subplots(2, 3, figsize=(6, 3),
                            sharex=True, sharey=True,
                            layout="constrained"
    )

    scatter_with_corr(df, "chi_base_u10_spatial", "chi_samp_u10_spatial", axs[0, 0], **scatter_kws)
    scatter_with_corr(df, "chi_base_u10_spatial", "chi_ht_u10_spatial", axs[1, 0], **scatter_kws)

    scatter_with_corr(df, "chi_base_tp_spatial", "chi_samp_tp_spatial", axs[0, 1], **scatter_kws)
    scatter_with_corr(df, "chi_base_tp_spatial", "chi_ht_tp_spatial", axs[1, 1], **scatter_kws)

    scatter_with_corr(df, "chi_base_mslp_spatial", "chi_samp_mslp_spatial", axs[0, 2], **scatter_kws)
    scatter_with_corr(df, "chi_base_mslp_spatial", "chi_ht_mslp_spatial", axs[1, 2], **scatter_kws)

    for ax in axs.flat:
        ax.set_xlabel("χ(0.8) - ERA5")
        ax.label_outer()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    axs[0, 0].set_title("u10", y=0.85, fontweight='bold')
    axs[0, 1].set_title("tp", y=0.85, fontweight='bold')
    axs[0, 2].set_title("mslp", y=0.85, fontweight='bold')
    axs[0,0].set_ylabel("hazGAN")
    axs[1,0].set_ylabel("H&T (2004)")

    caption = """Extremal coefficient χ(0.8) between 1000 randomly sampled pairs of locations for wind speed (u10),
    precipitation (tp), and mean sea level pressure (mslp). HT2004 refers to the method of Heffernan and Tawn (2004),
    where the the first member of each location pair is used as the conditioning variable. Both HT2004 and
    hazGAN generate 914 samples for each pair (corresponding to 500 years of events)."""
    print(caption)

    fig.savefig(figdir / "extcorrs_spatial.png", dpi=300, transparent=True)

    # %% plot multivariate extremal correlations
    fig, axs = plt.subplots(2, 3, figsize=(6, 3),
                            sharex=True, sharey=True,
                            layout="constrained"
    )

    scatter_with_corr(df, "chi_base_u10_tp", "chi_samp_u10_tp", axs[0, 0], **scatter_kws)
    scatter_with_corr(df, "chi_base_u10_tp", "chi_ht_u10_tp", axs[1, 0], **scatter_kws)

    scatter_with_corr(df, "chi_base_u10_mslp", "chi_samp_u10_mslp", axs[0, 1], **scatter_kws)
    scatter_with_corr(df, "chi_base_u10_mslp", "chi_ht_u10_mslp", axs[1, 1], **scatter_kws)

    scatter_with_corr(df, "chi_base_tp_mslp", "chi_samp_tp_mslp", axs[0, 2], **scatter_kws)
    scatter_with_corr(df, "chi_base_tp_mslp", "chi_ht_tp_mslp", axs[1, 2], **scatter_kws)

    for ax in axs.flat:
        ax.set_xlabel("χ(08) - ERA5")
        ax.label_outer()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    axs[0, 0].set_title("u10 v tp", y=0.85, fontweight='bold')
    axs[0, 1].set_title("u10 v mslp", y=0.85, fontweight='bold')
    axs[0,2].set_title("tp v mslp", y=0.85, fontweight='bold')
    axs[1,0].set_ylabel("H&T (2004)")
    axs[0,0].set_ylabel("hazGAN")

    fig.savefig(figdir / "extcorrs_multiv.png", dpi=300, transparent=True)
# %%