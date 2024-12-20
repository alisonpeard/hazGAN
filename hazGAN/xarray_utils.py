import numpy as np
import xarray as xr

def make_grid(ds, x='lon', y='lat', varname='x') -> xr.Dataset:
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=varname)
    h, w = ds.dims[y], ds.dims[x]
    grid = np.arange(0, h * w, 1).reshape(h, w)
    grid = xr.DataArray(
        grid, dims=[y, x], coords={y: ds[y][::-1], x: ds[x]}
    )
    ds['grid'] = grid
    return ds