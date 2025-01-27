# %%
import os
from environs import Env
import numpy as np
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

from hazGAN.utils import res2str
# %%
WINDTHRESHOLD = -float('inf')  #15 for storms
DOMAIN        = "gumbel" # ["uniform", "gumbel"]
EPS           = 1e-6

def apply_colormap(grayscale_array, colormap_name='Spectral_r'):
    normalized = grayscale_array.astype(float) / 255
    colormap = plt.get_cmap(colormap_name)
    colored = colormap(normalized)
    rgb_uint8 = np.uint8(colored[..., :3] * 255)
    return rgb_uint8


def create_image_grid(image_paths, grid_size=(32, 32), output_path="grid.png"):
    with Image.open(image_paths[0]) as img:
        tile_width, tile_height = img.size
        
    total_width = tile_width * grid_size[1]
    total_height = tile_height * grid_size[0]
    
    output_img = Image.new('RGB', (total_width, total_height), 'white')
    
    n_images = min(len(image_paths), grid_size[0] * grid_size[1])
    
    for idx in range(n_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        x = col * tile_width
        y = row * tile_height
        
        with Image.open(image_paths[idx]) as img:
            output_img.paste(img, (x, y))
            
    output_img.save(output_path)
    print(f"Grid saved to: {output_path}")
    return output_img

RES  = (64, 64)
CMAP = "Spectral_r"

env = Env()
env.read_env(recurse=True)
traindir = env.str("TRAINDIR")
traindir = os.path.join(traindir, res2str(RES))
os.listdir(traindir)
# %%
ds = xr.open_dataset(os.path.join(traindir, 'data.nc'))
ds['windmax'] = ds.sel(field='u10').anomaly.max(dim=['lon', 'lat'])
mask = (ds['windmax'] > WINDTHRESHOLD).values
idx  = np.where(mask)[0]
ds   = ds.isel(time=idx)

# %%
ds.isel(time=0, field=0).uniform.plot(cmap=CMAP)
ds.isel(time=0, field=0).uniform
ds = ds.fillna(1.) #! TEMPORARY

print(f"\nFound {ds.time.size} images with maximum wind exceeding {WINDTHRESHOLD} m/s")

# %% Make JPEGS of percentiles
winddir = os.path.join(traindir, 'images', DOMAIN, "wind")
stormdir = os.path.join(traindir, 'images', DOMAIN, "storm")
os.makedirs(winddir, exist_ok=True)
os.makedirs(stormdir, exist_ok=True)

nimgs = ds.time.size
array = ds.uniform.values
array = np.flip(array, axis=1) # flip latitude

if not ((array.max() <= 1.) and (array.min() >= 0.)):
        raise ValueError("Percentiles not in [0,1] range")

assert array.shape[1:] == (64, 64, 3), f"Unexpected shape: {array.shape}"

if DOMAIN == "gumbel":
    array = np.clip(array, EPS, 1-EPS) # Avoid log(0)
    array = -np.log(-np.log(array))
    array_min = np.min(array, axis=(0, 1, 2), keepdims=True)
    array_max = np.max(array, axis=(0, 1, 2), keepdims=True)
    n = len(array)

    # scale to (0, 1)
    array = (array - array_min) / (array_max - array_min)
    array = (array * (n - 1) + 1) / (n + 1)
    # original = ((scaled * (n+1) - 1) / (n-1)) * (max - min) + min

    print("Range:", array.min(), array.max())
    print("Shape:", array_min.shape, array_max.shape)

    stats_path = os.path.join(winddir, "..", "image_stats.npz")
    np.savez(stats_path, min=array_min, max=array_max, n=n)

for i in range(nimgs):
    arr = array[i]
    arr = np.uint8(arr * 255)
    
    first_channel = arr[..., 0]
    colored_array = apply_colormap(first_channel)  # Try different colormaps!
    colored_img = Image.fromarray(colored_array)
    output_path = os.path.join(winddir, f"wind_{i}.png")
    colored_img.save(output_path)

    img = Image.fromarray(arr, 'RGB')
    output_path = os.path.join(stormdir, f"storm_{i}.png")
    img.save(output_path)

    # Optional: verify saved image
    test_load = Image.open(output_path)

storm_paths = sorted(glob.glob(os.path.join(stormdir, "storm_*.png")))
wind_paths = sorted(glob.glob(os.path.join(winddir, "wind_*.png")))

# Create grids
create_image_grid(storm_paths, (8, 8), os.path.join(stormdir, "..", "percentiles_storm.png"))
create_image_grid(wind_paths, (8, 8), os.path.join(winddir, "..", "percentiles_wind.png"))

# %% Quantiles
winddir = os.path.join(traindir, 'images', "anomaly", "wind")
stormdir = os.path.join(traindir, 'images', "anomaly", "storm")
os.makedirs(winddir, exist_ok=True)
os.makedirs(stormdir, exist_ok=True)

nimgs = ds.time.size

for i in range(nimgs):
    array = ds.isel(time=i).anomaly.values
    array = np.flip(array, axis=0)

    if not ((array.max() <= 1.) and (array.min() >= 0.)):
        print("WARNING: Data not in [0,1] range, normalizing...")
        minima = np.min(array, axis=(0, 1), keepdims=True)
        maxima = np.max(array, axis=(0, 1), keepdims=True)
        array  = (array - minima) / (maxima - minima)

    assert array.shape== (64, 64, 3), f"Unexpected shape: {array.shape}"

    array = np.uint8(array * 255)

    first_channel = array[..., 0]
    colored_array = apply_colormap(first_channel)  # Try different colormaps!
    colored_img = Image.fromarray(colored_array)
    output_path = os.path.join(winddir, f"wind_{i}.png")
    colored_img.save(output_path)

    img = Image.fromarray(array, 'RGB')
    output_path = os.path.join(stormdir, f"storm_{i}.png")
    img.save(output_path)

    # Optional: verify saved image
    test_load = Image.open(output_path)

storm_paths = sorted(glob.glob(os.path.join(stormdir, "storm_*.png")))
wind_paths = sorted(glob.glob(os.path.join(winddir, "wind_*.png")))

print("{} storms processed".format(len(storm_paths)))

# Create grids
create_image_grid(storm_paths, (8, 8), os.path.join(stormdir, "..", "quantiles_storm.png"))
create_image_grid(wind_paths, (8, 8), os.path.join(winddir, "..", "quantiles_wind.png"))
# %%