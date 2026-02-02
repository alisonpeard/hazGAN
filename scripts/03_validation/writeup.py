"""
Make summary tables for each point of comparison.
"""
# %%
import numpy as np
import pandas as pd
from environs import Env
from pathlib import Path
from scipy.stats import wasserstein_distance

env = Env()
env.read_env()

figdir = Path(env.str("FIG_DIR"))

# %% Discuss extremal dependence results
wd = figdir / "corr-fields" 
domains = ["gaussian", "gumbel", "uniform"]

pearsoncols = ["pearson01", "pearson02", "pearson12"]
extremalcols = ["extremal01", "extremal02", "extremal12"]
headers = ["mae (best)", "mae (avg)", "r (best)", "r (avg)"]

results_r = {}
results_θ = {}

for domain in domains:
    statspath = wd / domain / "summary.csv"
    df = pd.read_csv(statspath, index_col=0)

    best_mae_r = df.loc["mae", pearsoncols].min()
    best_mae_θ = df.loc["mae", extremalcols].min()

    avg_mae_r = df.loc["mae", pearsoncols].mean()
    avg_mae_θ = df.loc["mae", extremalcols].mean()

    best_r_r = df.loc["pearson", pearsoncols].max()
    best_r_θ = df.loc["pearson", extremalcols].max()

    avg_r_r = df.loc["pearson", pearsoncols].mean()
    avg_r_θ = df.loc["pearson", extremalcols].mean()

    results_r[domain] = [best_mae_r, avg_mae_r, best_r_r, avg_r_r]
    results_θ[domain] = [best_mae_θ, avg_mae_θ, best_r_θ, avg_r_θ]

results_r = pd.DataFrame.from_dict(results_r, orient="index", columns=headers)
results_θ = pd.DataFrame.from_dict(results_θ, orient="index", columns=headers)

results = pd.concat([results_r, results_θ], keys=["pearson", "extremal"], axis=0)
results.to_csv(wd / "summary.csv")

# %% Discuss overall storm distributions
wd = figdir / "saffir-simpson" 
domains = ["gaussian", "gumbel", "uniform"]

headers = ["value", "train", "uniform", "gaussian", "gumbel",
           "Δuniform", "Δgaussian", "Δgumbel",
           "%uniform", "%gaussian", "%gumbel"]

dfs = []
for domain in domains:
    statspath = wd / f"{domain}.csv"
    df = pd.read_csv(statspath, index_col=0)
    df = df.reset_index()
    df["metric"] = df["metric"].replace({
        "fake": domain,
        "difference": f"Δ{domain}",
        "%difference": f"%{domain}"
    })
    dfs.append(df)

df = pd.concat(dfs, axis=0)
df = df.drop_duplicates(subset="metric")
df = df.set_index("metric").T
df = df[headers]
df.to_csv(wd / "summary.csv")
df

# %% kl divergence from 
"""Note this will not account for extrapolation (this is good)."""

def perc_to_float(s:str):
    if isinstance(s, float):
        return s
    return float(s.replace("%", "")) / 100.

dfkl = df.copy()
p = dfkl["train"].apply(perc_to_float)
p = np.clip(p, 1e-6, None)
p = p / p.sum()


results = {}

for domain in domains:
    
    q = dfkl[domain].apply(perc_to_float)
    q = np.clip(q, 1e-6, None)
    q = q / q.sum()
    ratio = (p / q)
    kl = (p * np.log(ratio))
    results[domain] = kl.sum()

dfkl = pd.DataFrame.from_dict(results, orient="index", columns=["kl divergence"])
dfkl.to_csv(wd / "kl_divergence.csv")

# %% jensen-shannon divergence
results = {}
for domain in domains:
    p = df["train"].apply(perc_to_float).fillna(0).values
    q = df[domain].apply(perc_to_float).fillna(0).values
    
    p = p / p.sum()
    q = q / q.sum()
    
    # M is the 'average' distribution
    m = 0.5 * (p + q)
    
    # JS calculation: 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
    kl_qm = np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))
    
    results[domain] = 0.5 * kl_pm + 0.5 * kl_qm

dfjs = pd.DataFrame.from_dict(results, orient="index", columns=["js divergence"])
dfjs.to_csv(wd / "js_divergence.csv")
# %% wasserstein distance
def extract_pos(s):
    return float(s.replace("cat-", ""))

#! hardcoded
positions = df["value"].astype(float).replace(float("nan"), 60.).values

results = {}
for domain in domains:
    p = df["train"].apply(perc_to_float).fillna(0).values
    q = df[domain].apply(perc_to_float).fillna(0).values
    
    p = p / p.sum()
    q = q / q.sum()
    
    results[domain] = wasserstein_distance(positions, positions, p, q)

dfwd = pd.DataFrame.from_dict(results, orient="index", columns=["wasserstein distance"])
dfwd.to_csv(wd / "wasserstein_distance.csv")
dfwd

# %%
dfall = pd.concat([dfkl, dfjs, dfwd], axis=1).T
dfall.to_csv(wd / "metrics.csv")
# %%

wd = figdir / "wasserstein" 
domains = ["gaussian", "gumbel", "uniform"]

dfs = []
for domain in domains:
    statspath = wd / f"{domain}.csv"
    df = pd.read_csv(statspath, index_col=0)
    df = df.reset_index()
    df["metric"] = df["metric"].replace({
        "fake": domain,
    })
    df["margins"] = domain
    dfs.append(df)

df = pd.concat(dfs, axis=0)
df.groupby(["metric", "margins"]).mean()
# %%
