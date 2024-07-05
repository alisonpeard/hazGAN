"""
Adapted from: https://github.com/alisonpeard/unops_bangladesh/blob/main/python/scripts/output1__intersection-snail.py#L109
"""
# %%
import os
import pandas
import geopandas as gpd
import snail
import rasterio

# %%
raster = '/Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/expected_annual_damages.tif'
polygon = '/Users/alison/Documents/DPhil/paper1.nosync/results/mangroves/mangroves.geojson'
# %%
mangroves = gpd.read_file(polygon)
mangroves
# %% ---- Get transforms ----
from collections import namedtuple
Transform = namedtuple('Transform', ['crs', 'width', 'height', 'transform'])
with rasterio.open(raster) as src:
    t = Transform(src.crs, src.width, src.height, tuple(src.transform))
t
# %%
def explode_multi(df):
    """Explode any MultiPoint, MultiLineString, MultiPolygon geometries into multiple rows with single geometries."""
    items = []
    geoms = []
    for item in df.itertuples(index=False):
        if item.geometry.geom_type in ('MultiPoint', 'MultiLineString', 'MultiPolygon'):
            for part in item.geometry:
                items.append(item._asdict())
                geoms.append(part)
        else:
            items.append(item._asdict())
            geoms.append(item.geometry)

    df = gpd.GeoDataFrame(items, crs=df.crs, geometry=geoms)
    return df

areas = explode_multi(mangroves)

# %%
import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import polygonize
from snail.core.intersections import split_polygon

def set_precision(geom, precision):
    """Set geometry precision"""
    geom_mapping = mapping(geom)
    geom_mapping["coordinates"] = np.round(
        np.array(geom_mapping["coordinates"]), precision)
    return shape(geom_mapping)


def split_area_df(df, t):
    # split
    core_splits = []
    for area in df.itertuples():  # tqdm , total=len(df)
        # split area
        splits = split_polygon(
            area.geometry,
            t.width,
            t.height,
            t.transform
        )
        # round to high precision (avoid floating point errors)
        splits = [set_precision(s, 9) for s in splits]
        # to polygons
        splits = list(polygonize(splits))
        # add to collection
        for s in splits:
            s_dict = area._asdict()
            del s_dict['Index']
            s_dict['geometry'] = s
            core_splits.append(s_dict)
    sdf = gpd.GeoDataFrame(core_splits)
    sdf.crs = t.crs
    return sdf

# %% process areas
crs_df = areas.to_crs(t.crs)
crs_df = split_area_df(crs_df, t)
# %%
from snail.core.intersections import get_cell_indices
def get_indices(geom, t):
    """Get the indices of the raster cell in the middle of the geometry."""
    x, y = get_cell_indices(
        geom,
        t.width,
        t.height,
        t.transform)
    # wrap around to handle edge cases
    x = x % t.width
    y = y % t.height
    return (x, y)

crs_df[f'cell_index'] = crs_df.geometry.apply(lambda geom: get_indices(geom, t))
# %%
areas = crs_df.to_crs(areas.crs)
# %% associate raster
def associate_raster(df, key, fname, cell_index_col='cell_index', band_number=1):
    with rasterio.open(fname) as dataset:
        band_data = dataset.read(band_number)
        df[key] = df[cell_index_col].apply(lambda i: band_data[i[1], i[0]])

associate_raster(areas, 'EAD', raster)
# %%
def split_index_column(df, prefix):
    df[f'{prefix}_x'] = df[prefix].apply(lambda i: i[0])
    df[f'{prefix}_y'] = df[prefix].apply(lambda i: i[1])
    print("split_index_column_done")
    return df

areas = split_index_column(areas, f'cell_index')
areas.drop(columns=f'cell_index', inplace=True)
# %%
import matplotlib.pyplot as plt

areas.plot('EAD', cmap='YlOrRd')
plt.xlim(88, 90)
plt.ylim(21, 23)
# %%
areas
# %%
areas['EAD'].hist()
# %%
