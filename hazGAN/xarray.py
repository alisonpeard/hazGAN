import xarray as xr

def make_grid(dataset:xr.Dataset, x="lon", y="lat") -> xr.Dataset:
    """Helper function for grid-based indexing"""
    dataset = dataset.stack(grid=[y, x])
    return dataset