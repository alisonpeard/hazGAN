# %%
import os
import sys
import numpy as np
import xarray as xr
import xagg as xa
import geopandas as gpd
from joblib import load

from ..constants import bay_of_bengal_crs
from . import MangroveDamage

class mangroveDamageModel(object):
    def __init__(self, modelpath=None) -> None:
        modelpath = modelpath or os.path.join(os.path.dirname(__file__), 'damagemodel.pkl')
        self.modelpath = modelpath
        self.model = None
        self.metrics = None
        self.load_model()
        #return self

    def load_model(self) -> None:
        """Load a pickled sklearn model"""
        path = sys.path.copy()
        sys.path.insert(0, os.path.dirname(__file__))
        with open(self.modelpath, 'rb') as f:
            self.model = load(f)
            self.metrics = self.model.metrics
        sys.path = path

    def __str__(self) -> str:
        out = '\n'.join(["Mangrove Damage Model",
                        "---------------------",
                        "Model: Logistic Regression",
                        "Trained on: IBTrACS",
                        "Predicts: P[Δ(NDVI) ≤ -20%]",
                        "\nMetrics", "-------",
                        str(self.metrics)]
                        )
        return out

    def damage_ufunc(self, X:np.ndarray) -> np.ndarray:
        X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict(
        self,
        ds:xr.DataArray,
        vars_:list,
        regressors=['u10', 'tp'],
        first_dim='sample',
        reduce_dim='field',
        ) -> xr.DataArray:
        """
        Calculate mangrove damages using a pretrained scikit-learn model.
        
        https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
        """
        predictions = ds.copy()
        ds = ds.sel(field=regressors)
        ds = ds.copy().chunk({first_dim: "auto"})
        for var in vars_:
            predictions[f"{var}_damage"] = xr.apply_ufunc(
                self.damage_ufunc,
                ds[var],
                input_core_dims=[[reduce_dim]],  # apply ufunc along
                output_core_dims=[[]],           # dimensions results are in
                exclude_dims=set((reduce_dim,)), # dimensions allowed to change size, must be set!
                vectorize=True,                  # loop over non-core dimensions
                dask="parallelized",
                output_dtypes=[float]
            )
        print("Computing chunked predictions")
        return predictions.compute()

    def intersect_mangroves_with_grid(
            self,
            mangroves:gpd.GeoDataFrame,
            grid:xr.DataArray,
            crs=4326, #bay_of_bengal_crs,
            crs_out=None
            ) -> xr.Dataset:
        """Use xagg to intersect mangroves with damage fields"""
        # calculate intersections
        mangroves = mangroves.to_crs(crs)
        # grid   = grid.rio.reproject(crs)
        weightmap = xa.pixel_overlaps(grid, mangroves)

        # calculate overlaps
        mangroves_gridded = weightmap.agg
        mangroves_gridded['npix'] = mangroves_gridded['pix_idxs'].apply(len)
        mangroves_gridded['rel_area'] = mangroves_gridded['rel_area'].apply(lambda x: np.squeeze(x, axis=0))
        mangroves_gridded = mangroves_gridded.explode(['rel_area', 'pix_idxs'])

        # sum all relative mangrove areas in the same pixel
        mangroves_gridded['area'] = mangroves_gridded['area'] * mangroves_gridded['rel_area']
        mangroves_gridded = mangroves_gridded.groupby('pix_idxs').agg({'area': 'sum', 'coords': 'first'})

        # convert pd.DataFrame to xarray.Dataset
        lons = weightmap.source_grid['lon'].values
        lats = weightmap.source_grid['lat'].values
        mangroves_gridded = mangroves_gridded.reset_index()
        mangroves_gridded['lon'] = mangroves_gridded['pix_idxs'].apply(lambda j: lons[j])
        mangroves_gridded['lat'] = mangroves_gridded['pix_idxs'].apply(lambda i: lats[i])
        mangroves_gridded['lon'] = mangroves_gridded['lon'].astype(float)
        mangroves_gridded['lat'] = mangroves_gridded['lat'].astype(float)
        mangroves_gridded['area'] = mangroves_gridded['area'].astype(float)
        mangroves_gridded['area'] = mangroves_gridded['area'] * 1e-6 # convert to sqkm
        mangroves_gridded = mangroves_gridded.set_index(['lat', 'lon'])[['area']]
        mangroves_gridded = xr.Dataset.from_dataframe(mangroves_gridded)

        # if crs_out:
        #     mangroves_gridded = mangroves_gridded.rio.set_crs(crs_out)
        # if plot:
        #     mangroves_gridded.area.plot(cmap="Greens", cbar_kwargs={'label': 'Mangrove damage [km²]'})
        return mangroves_gridded
