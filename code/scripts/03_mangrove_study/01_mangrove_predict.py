# %%
import numpy as np
import os
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt

from utils import mangroveDamageModel
from utils.analysis import load_samples

env = Env()
env.read_env()

# %% load generated data
THRESHOLD = 15. # None for all storms
TYPE = "trunc-1_0"
MODEL     = 30 
MODEL     = str(MODEL).zfill(5) if isinstance(MODEL, int) else MODEL
MONTH     = 9 #"September"
NYEARS    = 500

samples_dir = env.str("SAMPLES_DIR")
data_dir    = env.str("DATADIR")
train_dir   = env.str("TRAINDIR")

samples_dir = os.path.expanduser(samples_dir)
data_dir    = os.path.expanduser(data_dir)
train_dir   = os.path.expanduser(train_dir)

# samples = load_samples(samples_dir, data_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE)
samples = load_samples(samples_dir, data_dir, train_dir, MODEL, threshold=THRESHOLD, sampletype=TYPE, ny=NYEARS)
data    = xr.open_dataset(os.path.join(train_dir, "data.nc"))
nobs    = data.sizes['time']
nyears = len(np.unique(data['time.year']))

medians = data['medians']
medians["month"] = medians["time.month"]
month_time_mask = medians["month"] == MONTH
medians = medians.isel(time=month_time_mask)
medians = medians.mean(dim='time').values

# get event sets
train       = samples['training']['data'] + medians
fake        = samples['samples']['data'] + medians
independent = samples['assumptions']['independent'] + medians
dependent   = samples['assumptions']['dependent'] + medians

dependent_uniform = samples['assumptions']['dependent_u']
dependent_rp      = samples['assumptions']['dependent_rp']

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

trainda = to_xarray(train)
fakeda = to_xarray(fake)
independentda = to_xarray(independent)
dependentda = to_xarray(dependent)
dependent_uniformda = to_xarray(dependent_uniform)
dependent_rpda = to_xarray(dependent_rp)

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

# predict mangrove damages
model = mangroveDamageModel()

# %%
if True:

    train_damages = model.predict(train, ["train"])
    train_damages = train_damages.rename({"train_damage": "damage_prob"})
    train_damages.to_netcdf(os.path.join(data_dir, "mangroves", "train_damages.nc"))
if False: # to avoid repeating unnecessary calculations
    fake_damages = model.predict(fake, ["fake"])
    fake_damages = fake_damages.rename({"fake_damage": "damage_prob"})
    fake_damages.to_netcdf(os.path.join(data_dir, "mangroves", "fake_damages.nc"))
    

    # predict dependent and independent damages
    dependent_damages = model.predict(dependent, ["dependent"])
    dependent_damages = dependent_damages.rename({"dependent_damage": "damage_prob"})
    dependent_damages.to_netcdf(os.path.join(data_dir, "mangroves", "dependent_damages.nc"))

    independent_damages = model.predict(independent, ["independent"])
    independent_damages = independent_damages.rename({"independent_damage": "damage_prob"})
    independent_damages.to_netcdf(os.path.join(data_dir, "mangroves", "independent_damages.nc"))

# %%