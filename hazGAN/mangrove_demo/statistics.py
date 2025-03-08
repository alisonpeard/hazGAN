import xarray as xr
from sklearn.metrics import auc


def calculate_hazard_map(rp:int,
                         damages:xr.Dataset,
                         yearly_rate:float,
                         var:str) -> xr.Dataset:
    # totals = damages[var].sum(dim=['lat', 'lon']).to_dataset()
    N = damages[var].sizes['sample']
    ranks = damages[var].rank(dim=['lat', 'lon'])
    damages['ecdfs'] = ranks / (1 + N)
    damages['exceedance'] = 1 - damages['ecdfs']

    # now we need to find the value of the variable that corresponds to the exceedence probability
    exceedance_annual = 1 - (rp / (yearly_rate * N))

    # interpolate each (gridcell, field) to find the value of the variable that corresponds to the exceedence probability
    hazard_map = damages[var].interp(exceedance=exceedance_annual)

    return hazard_map


def calculate_total_return_periods(damages:xr.Dataset,
                                   yearly_rate:float,
                                   var:str) -> xr.Dataset:
    totals = damages[var].sum(dim=['lat', 'lon']).to_dataset()
    N = totals[var].sizes['sample']
    rank = totals[var].rank(dim='sample')
    totals['exceedence_prob'] = 1 - rank / (1 + N)
    totals['return_period'] = 1 / (yearly_rate * totals['exceedence_prob'])
    totals = totals.sortby('return_period')
    return totals


def calculate_eads(var, damages:xr.Dataset, yearly_rate:int) -> xr.Dataset:
    """https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html"""
    def auc_ufunc(x, y):
        x = sorted(x)
        y = sorted(y)
        out = auc(x, y)
        return out

    basename = var.split('_')[0]
    damages = damages.copy()
    exceedence_prob = 1 - (damages[var].rank(dim='sample') / (1 + damages[var].sizes['sample']))

    annual_exceedence_prob = yearly_rate * exceedence_prob
    return_period = 1 / annual_exceedence_prob
    damages[f'{basename}_annual_exceedence_prob'] = annual_exceedence_prob
    damages[f'{basename}_return_period'] = return_period

    EADs = xr.apply_ufunc(
        auc_ufunc,
        damages[f'{basename}_annual_exceedence_prob'],
        damages[var],
        input_core_dims=[['sample'], ['sample']],
        output_core_dims=[[]],
        exclude_dims=set(('sample',)), # dimensions allowed to change size, must be set!
        vectorize=True,                # loop over non-core dimensions,
        dask="parallelized",
        output_dtypes=[float]
        )
    
    damages[f'{basename}_EAD'] = EADs
    return damages


