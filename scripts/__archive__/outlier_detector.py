"""
Some unusual wind fields show up in the northwest corner of the Bay of Bengal. Almost
resembling the "rain bombs" which are well-documented in ERA5. Additionally, some 
have values exceeding 50mps which is considered unrealistic for the ~30km ERA5
grid resolution. 

As a solution, the most severe known example of this is taken as a template (1D) matrix
and the Frobenius norm computed with the wind field every day of the dataset. This method
finds three more examples of the "wind bomb" and a file with the offending timestamps
is exported to the training data folder.
"""
# %% load the environment
import os
import glob
import numpy  as np
import pandas as pd
import xarray as xr
from environs import Env
import matplotlib.pyplot as plt

THRESHOLD = 0.8 # to start

def frobenius(test:np.ndarray, template:np.ndarray) -> float:
    similarity = np.sum(template * test) / (np.linalg.norm(template) * np.linalg.norm(test))
    return similarity


if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)  # read .env file, if it exists
    era5dir = env.str('ERA5DIR')
    outdir = env.str('TRAINDIR')
    files = glob.glob(os.path.join(era5dir, "*.nc"))

    # load data and process
    ds = xr.open_mfdataset(files)
    ds['wind'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
    ds['maxwind'] = ds['wind'].max(dim=['lat', 'lon'])

    # %% sort by maximum wind
    ds = ds.sortby('maxwind', ascending=False)
    fig, axs = plt.subplots(8, 8)
    for i, ax in enumerate(axs.ravel()):
        ds.isel(time=i).wind.plot(ax=ax, add_colorbar=False)
        ax.axis('off')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

    # %% Frobenius inner product
    template = ds.isel(time=0).wind.values
    test_matrices = ds.wind.values

    similarities = []
    for i in range(len(test_matrices)):
        test_matrix = test_matrices[i, ...]
        similarity = frobenius(test_matrix, template)
        similarities.append(similarity)
    similarities = np.array(similarities)

    ranks = np.argsort(similarities)[::-1]

    fig, axs = plt.subplots(8, 8)
    for i, ax in enumerate(axs.ravel()):
        ds.isel(time=ranks[i]).wind.plot(ax=ax, add_colorbar=False)
        ax.axis('off')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

        print(ds.isel(time=ranks[i]).time)
        print(ds.isel(time=ranks[i]).maxwind.values)
    plt.show()

    # %% expoer timestamps to training folder
    i = 0
    similarity_dict = {}
    for time, maxwind, similarity in zip(ds.time.values, ds.maxwind.values, similarities):
        similarity_dict[i] = [time, maxwind, similarity]
        i += 1

    similarity_df = pd.DataFrame.from_dict(similarity_dict, orient='index',
                                           columns=['time', 'max_wind', 'frobenius'])
    
    # personally only think top three count so...
    similarity_df = similarity_df.iloc[:3, :]
    similarity_df.to_csv(os.path.join(outdir, 'outliers.csv'))
# %%

