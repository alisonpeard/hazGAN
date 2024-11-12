# %%
import os
from osgeo import gdal
import netCDF4 as nc
from environs import Env
import numpy as np
import xarray as xr


def inspect_gmt_coords(input_path):
    """
    Inspect the coordinate system of the GMT file.
    """
    ds = nc.Dataset(input_path, 'r')
    print("\nFile Metadata:")
    for attr in ds.ncattrs():
        print(f"{attr}: {ds.getncattr(attr)}")
    
    print("\nDimensions:", dict(ds.dimensions))
    print("\nVariables:")
    for var in ds.variables:
        print(f"\n{var}:")
        for attr in ds.variables[var].ncattrs():
            print(f"  {attr}: {ds.variables[var].getncattr(attr)}")
    
    # Print coordinate ranges
    x_range = ds.variables['x_range'][:]
    y_range = ds.variables['y_range'][:]
    dimension = ds.variables['dimension'][:]
    spacing = ds.variables['spacing'][:]
    
    print("\nCoordinate Information:")
    print(f"X range: {x_range} (spacing: {spacing[0]})")
    print(f"Y range: {y_range} (spacing: {spacing[1]})")
    print(f"Dimensions: {dimension}")
    
    ds.close()
    return x_range, y_range, dimension, spacing


def gmt_grd_to_xarray(input_path, output_path, lat_bounds=None, lon_bounds=None):
    """
    Convert GMT-format GRD file to a standard netCDF/xarray format with optional subsetting.
    Handles potential coordinate system issues.
    """
    # First inspect the coordinates
    x_range, y_range, dimension, spacing = inspect_gmt_coords(input_path)
    
    # Open the GMT format dataset
    ds = nc.Dataset(input_path, 'r')
    
    # Create coordinate arrays with explicit spacing
    # Adjust if coordinates are not in expected order
    lons = np.arange(x_range[0], x_range[1] + spacing[0], spacing[0])
    lats = np.arange(y_range[0], y_range[1] + spacing[1], spacing[1])
    
    # Verify the dimensions match what we expect
    if len(lons) != dimension[0] or len(lats) != dimension[1]:
        print("Warning: Calculated dimensions don't match file dimensions")
        print(f"Calculated: {len(lons)}x{len(lats)}, File: {dimension[0]}x{dimension[1]}")
        # Adjust arrays to match file dimensions
        lons = np.linspace(x_range[0], x_range[1], dimension[0])
        lats = np.linspace(y_range[0], y_range[1], dimension[1])
    
    # Reshape the flat array into 2D grid
    elevation = np.array(ds.variables['z'][:]).reshape(dimension[1], dimension[0])
    
    # Check if we need to flip the latitude axis
    if lats[0] > lats[-1]:
        print("Flipping latitude axis to be ascending")
        lats = lats[::-1]
        elevation = elevation[::-1]
    
    # Create xarray Dataset with explicit coordinates
    da = xr.DataArray(
        data=elevation,
        dims=['latitude', 'longitude'],
        coords={
            'latitude': lats,
            'longitude': lons
        },
        attrs={
            'long_name': 'elevation',
            'units': 'meters',
            'description': 'ETOPO1 bedrock elevation',
            'crs': 'EPSG:4326'  # Explicitly set CRS
        }
    )
    
    ds_xr = da.to_dataset(name='elevation')
    
    # Add metadata
    ds_xr.attrs['title'] = ds.title if hasattr(ds, 'title') else 'ETOPO1 elevation data'
    ds_xr.attrs['source'] = ds.source if hasattr(ds, 'source') else 'Converted from GMT format'
    ds_xr.attrs['coordinate_reference_system'] = 'EPSG:4326'
    
    # Subset if bounds are provided
    if lat_bounds is not None and lon_bounds is not None:
        ds_xr = ds_xr.sel(
            latitude=slice(lat_bounds[0], lat_bounds[1]),
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )
    
    # Save to netCDF file
    ds_xr.to_netcdf(output_path)
    ds.close()
    
    return ds_xr


def plot_elevation(ds_xr, path=None):
    """
    Create a detailed plot of the elevation data.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Create figure with proper projection
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Plot elevation data
    plot = ds_xr.elevation.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='terrain',
        cbar_kwargs={'label': 'Elevation (m)'}
    )
    
    # Add coastlines and borders for reference
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Set title
    ax.set_title('ETOPO1 Elevation')
    
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches='tight')


# Example usage
if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)  # read .env file, if it exists
    datadir = env.str('DATADIR')
    mistral = env.str('MISTRAL')
    inpath = os.path.join(datadir, 'geo', 'etopo1', '1_arc_min', 'grd', 'ETOPO1_Bed_g_gdal.grd')
    outpath = os.path.join(mistral, 'hazGAN', 'constant-fields', 'etopo1.grd')

    lat_bounds = (10, 25)  # (min_lat, max_lat)
    lon_bounds = (80, 95)  # (min_lon, max_lon)

    ds = xr.open_dataset(inpath, engine="netcdf4")

    nx = 21601  # Longitude points
    ny = 10801  # Latitude points

    # Reshape with these exact dimensions
    z_2d = ds.z.values.reshape((ny, nx))
    z_2d = np.flip(z_2d, axis=0)

    # Create proper coordinate arrays at 1 arc-minute resolution
    lon = np.linspace(-180, 180, nx)
    lat = np.linspace(-90, 90, ny)

    # Create the new dataset
    ds_2d = xr.Dataset(
        data_vars={
            'z': (['lat', 'lon'], z_2d)
        },
        coords={
            'lon': ('lon', lon),
            'lat': ('lat', lat)
        },
        attrs=ds.attrs
    )

    print(ds_2d.z.shape)  # Should show (ny, nx)
    print(ds_2d)

    # Check the exact values
    print("Dimensions:", ds.dimension.values)
    print("X range:", ds.x_range.values)
    print("Y range:", ds.y_range.values)
    print("Z range:", ds.z_range.values)
    print("Spacing:", ds.spacing.values)
    print("Total points in z:", ds.z.size)

    # Also let's look at some z values
    print("\nFirst few z values:", ds.z.values[:10])
    print("Last few z values:", ds.z.values[-10:])

    if True:
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(
            figsize=(12, 8),
            subplot_kw={'projection': ccrs.PlateCarree()}
            )
        
        ds_2d.sel(lon=slice(80, 95), lat=slice(10, 25))['z'].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='terrain',
            cbar_kwargs={'label': 'Elevation (m)'}
            
            )
        
            # Add coastlines and borders for reference
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
    
        # Add gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    
        # Set title
        ax.set_title('ETOPO1 Elevation')

        figpath = os.path.join(mistral, 'hazGAN', 'etopo1-bay-of-bengal.png')
        fig.savefig(figpath, dpi=300, bbox_inches='tight')
        print('saved figure as', figpath)

    