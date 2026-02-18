# %%
import numpy as np
import os
import xarray as xr
from environs import Env
# import matplotlib.pyplot as plt

from utils import mangroveDamageModel
from utils.analysis import load_samples
from pathlib import Path

env = Env()
env.read_env()

# %% load generated data
thresh = 15. # None for all storms
nyrs = 500
domain = "gaussian"
scaling = "rp10000"
# MODEL     = 30 
# MODEL     = str(MODEL).zfill(5) if isinstance(MODEL, int) else MODEL
month     = 9 #"September"

# samples_dir = env.str("SAMPLES_DIR")
# data_dir    = env.str("DATADIR")
# train_dir   = env.str("TRAINDIR")


# configure paths   
train_dir = Path(env.str("TRAINDIR"))
samples_dir = Path(env.str("SAMPLES_DIR")) / scaling / domain / "npy"
figdir = Path(env.str("FIG_DIR")) / "mangroves"
figdir.mkdir(parents=True, exist_ok=True)
mangrove_dir = Path(env.str("MANGROVE_DIR"))
medians_path = train_dir / "data.nc"

samples_dir = os.path.expanduser(samples_dir)
# data_dir    = os.path.expanduser(data_dir)
train_dir   = os.path.expanduser(train_dir)

# samples = load_samples(samples_dir, data_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE)
# samples = load_samples(samples_dir, data_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE, ny=NYEARS)
samples = load_samples(
        samples_dir, train_dir,
        threshold=thresh, nyrs=nyrs,
        domain=domain, scaling=scaling,
        make_benchmarks=True
)

# %%
data    = xr.open_dataset(os.path.join(train_dir, "data.nc"))
nobs    = data.sizes['time']
nyears = len(np.unique(data['time.year']))

medians = data['medians']
medians = medians.sel(month=month).values

# get event sets
x_trn  = samples['training']['x'] + medians
x_gen  = samples['samples']['x'] + medians
x_ind  = samples['independent']['x'] + medians
x_dep  = samples['dependent']['x'] + medians

u_dep  = samples['dependent']['u']
rp_dep = samples['dependent']['rp']

# %% Turn them all into netcdf
ref_data = xr.open_dataset(os.path.join(train_dir, "data.nc"))
coords = ref_data.coords
lat = coords['lat'].data
lon = coords['lon'].data

def to_xarray(array):
    da = xr.DataArray(array, coords=[
    ('sample', range(len(array))),
    ('lat', lat),
    ('lon', lon),
    ('field', ['u10', 'tp', 'mslp'])
    ])
    return da

trainda = to_xarray(x_trn)
fakeda = to_xarray(x_gen)
independentda = to_xarray(x_ind)
dependentda = to_xarray(x_dep)
dependent_uniformda = to_xarray(u_dep)
dependent_rpda = to_xarray(rp_dep)

# %% make a dataset using their names
train = xr.Dataset({'train': trainda})
fake = xr.Dataset({'fake': fakeda})
independent = xr.Dataset({'independent': independentda})

# %% dependent data needs some extra steps
dependent   = xr.Dataset({'dependent': dependentda})
dependent['uniform'] = dependent_uniformda
dependent['return_period'] = dependent_rpda

# %%
# convert return periods to 1-d array
assert np.isclose(dependent['uniform'].std(dim=['lat', 'lon']).max(), 0)
assert np.isclose(dependent['return_period'].std(dim=['lat', 'lon']).max(), 0)
dependent['uniform'] = dependent['uniform'].mean(dim=['lat', 'lon'])
dependent['return_period'] = dependent['return_period'].mean(dim=['lat', 'lon', 'field'])

# %% check for nans
for ds in [train, fake, independent, dependent]:
    assert ds.isnull().sum() == 0, "Nans found in dataset"

# %% predict mangrove damages
# sklearn: Trying to unpickle estimator LogisticRegression from version 1.3.1 when using version 1.8.0. This might lead to breaking code or invalid results.
model = mangroveDamageModel(modelpath=mangrove_dir / "damagemodel.pkl")

# %%
results_dir = mangrove_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

if False:
    train_damages = model.predict(train, ["train"])
    train_damages = train_damages.rename({"train_damage": "damage_prob"})
    train_damages.to_netcdf(os.path.join(results_dir, "train_damages.nc"))
if False: # to avoid repeating unnecessary calculations
    fake_damages = model.predict(fake, ["fake"])
    fake_damages = fake_damages.rename({"fake_damage": "damage_prob"})
    fake_damages.to_netcdf(os.path.join(results_dir, "fake_damages.nc"))
    
if True:
    # predict dependent and independent damages
    dependent_damages = model.predict(dependent, ["dependent"])
    dependent_damages = dependent_damages.rename({"dependent_damage": "damage_prob"})
    dependent_damages.to_netcdf(os.path.join(results_dir, "dependent_damages.nc"))

if False:
    independent_damages = model.predict(independent, ["independent"])
    independent_damages = independent_damages.rename({"independent_damage": "damage_prob"})
    independent_damages.to_netcdf(os.path.join(results_dir, "independent_damages.nc"))

# %%