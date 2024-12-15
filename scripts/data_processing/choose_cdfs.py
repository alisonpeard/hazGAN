""""
Compare the fits of the empirical and GPD distributions to the data.

==> Choose pure GPD for the data, it performs better on train and test.
"""

# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
from environs import Env
from hazGAN.utils import TEST_YEAR
from hazGAN.statistics import GenPareto, Empirical


def mape(x, y):
    if np.isclose(x, 0).any():
        x = x + 1e-6
    return np.mean(np.abs((x - y) / x)) * 100


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    
    storms = pd.read_parquet(os.path.join(datadir, "storms.parquet"))
    storms['time.u10'] = pd.to_datetime(storms['time.u10'])
    storms['time.tp'] = pd.to_datetime(storms['time.tp'])
    storms['time.mslp'] = pd.to_datetime(storms['time.mslp'])
    storms = storms[storms['time.u10'].dt.year != TEST_YEAR]
    storms_test = storms[storms['time.u10'].dt.year == TEST_YEAR]

    data = xr.open_dataset(os.path.join(datadir, "data.nc"))
    mask = data['time.year'] != TEST_YEAR
    test_mask = data['time.year'] == TEST_YEAR

    data_test = data.sel(time=test_mask)
    data = data.sel(time=mask)

    # big comparison of different fits
    differences = []
    for ds in [data, data_test]:
        difference_dict = {}
        for FIELD in ['u10', 'tp', 'mslp']:
            maxdiffs_emp = []
            maxdiffs_hybrid = []
            maxdiffs_gpd = []
            for CELL in range(1, 18*22):
                test = data.sel(field=FIELD)
                test = test.where(test['grid'] == CELL, drop=True)
                test

                # test forward <==> inverse
                scale = test.sel(param='scale')['params'].data.item()
                shape = test.sel(param='shape')['params'].data.item()
                loc   = test.sel(param='loc')['params'].data.item()
                x     = test['anomaly'].data.squeeze()

                gpd_fit = GenPareto(x, loc, scale, shape)
                fit  = Empirical(x)

                # pure empirical forward <==> inverse
                field_u = fit.forward(x)
                field_x = fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_emp.append(maxdiff)

                # pure GPD forward <==> inverse
                field_u = gpd_fit.forward(x)
                field_x = gpd_fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_gpd.append(maxdiff)

                # empirical + GPD forward <==> inverse
                field_u = fit.forward(x)
                field_x = gpd_fit.inverse(field_u)
                difference = x - field_x
                maxdiff = difference.max()
                maxdiffs_hybrid.append(maxdiff)
            
            difference_dict[FIELD] = {
                'empirical': np.sum(maxdiffs_emp),
                'gpd': np.sum(maxdiffs_gpd),
                'hybrid': np.sum(maxdiffs_hybrid)
                }
            
        differences.append(difference_dict)

    differences = {'train': differences[0], 'test': differences[1]}
    reform = {(outerKey, innerKey): values for outerKey, innerDict in differences.items() for innerKey, values in innerDict.items()}
    differences = pd.DataFrame(reform).T
# %%