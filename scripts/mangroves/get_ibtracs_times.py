"""
Augment Yu's data
    - Add Athanisou's slope data
    - Add ERA5 pressure and wind at landfall time at each patch (no more Holland)
To do:
    - Extract lifetime maximum wind instead of just the wind at the time of landfall
"""
#%%
import os
import pandas as pd
import geopandas as gpd
bob_crs = 24346

# %% new way to organise files
indir = '/Users/alison/Documents/DPhil/paper1.nosync/mangrove_data/v3__mine'
outdir = indir
infile1 = os.path.join(indir, 'data_with_slopes.gpkg')
infile2 = '/Users/alison/Documents/DPhil/data/ibtracs/ibtracs_since1980.csv'
outfile1 = os.path.join(outdir, 'data_with_ibtracs.csv')

# %% ----- Get storm times from IBTrACs ------
gdf = gpd.read_file(infile1)
ibtracs = pd.read_csv(infile2)
ibtracs = ibtracs.groupby(['SID']).agg({'NAME': 'first', 'ISO_TIME': list}).reset_index()
ibtracs = ibtracs[['NAME', 'ISO_TIME']].rename(columns={'ISO_TIME': 'times'})
gdf['stormName'] = gdf['stormName'].str.upper() 
gdf = pd.merge(gdf, ibtracs, left_on='stormName', right_on='NAME', how='inner')
gdf = gdf.groupby(['center_centerLat', 'center_centerLon', 'stormName']).agg({'NAME': 'first', 'times': 'first'}).reset_index()
gdf = gdf[['stormName', 'center_centerLat', 'center_centerLon', 'times']]
gdf.columns = ['storm', 'lat', 'lon', 'times']

gdf.to_csv(outfile1)
# %%
