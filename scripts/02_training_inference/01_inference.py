"""
This script searches the samples directory for generated images and the training directory
for training data and parameters to convert images back to original scale.
"""
# %%
# quick defaults
TRAINRES = 64
STEP     = 300
MODEL    = "00006-storms-low_shot-translation"
WD       = "/soge-home/projects/mistral/alison/data/stylegan/training-runs"

import os
import glob
import torch
import argparse
import numpy as np
import xarray as xr
from PIL import Image
from environs import Env
from numpy import asarray
import matplotlib.pyplot as plt
from hazGAN.statistics import invPIT
from torchvision.transforms.functional import resize


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


# script begins here
if __name__ == "__main__":
    print("\nReading environment variables...")
    # process args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', dest="WD", type=str, default=WD, help='Training runs directory.')
    parser.add_argument('--model', '-m', dest="MODEL", type=str, default=MODEL)
    parser.add_argument('--step', '-s', dest='STEP', type=int, default=STEP)
    parser.add_argument('--train-res', '-t', dest='TRAINRES', type=int, default=TRAINRES, help="Training data's original resolution.")
    parser.add_argument("--res", '-r', type=int, default=64, choices=[16, 32, 64, 128, 256, 512], help="Sample resolution")
    args = parser.parse_args()

    WD       = args.WD
    MODEL    = args.MODEL
    TRAINRES = args.TRAINRES
    STEP     = str(args.STEP).zfill(6)
    RES      = args.res
    args     = parser.parse_args()
    print(f"Making results images for resolution: {RES} pixels.")

    # source training data
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")
    datadir = os.path.join(datadir, f"{TRAINRES}x{TRAINRES}")

    # model run directories
    resultsdir = os.path.join(WD,  MODEL, "results")
    indir      = os.path.join(resultsdir, "samples")
    winddir    = os.path.join(resultsdir, "samples_single_field")
    griddir    = os.path.join(resultsdir, "grids")
    percdir    = os.path.join(winddir, "percentiles")
    quantdir   = os.path.join(winddir, "quantiles")

    os.makedirs(winddir, exist_ok=True)
    os.makedirs(percdir, exist_ok=True)
    os.makedirs(quantdir, exist_ok=True)
    os.makedirs(griddir, exist_ok=True)

    imlist = glob.glob(os.path.join(indir, "seed*.png"))
    print(f"Found {len(imlist)} images in {indir}.")

    # load generated images and convert to numpy
    print("Loading generated images...")
    samples = []
    for i in range(len(imlist)):
        impath = imlist[i]
        image = Image.open(os.path.join(indir, impath))
        array = asarray(image) / 255

        if not ((array.max() <= 1.) and (array.min() >= 0.)):
            print("WARNING: Data not in [0,1] range, normalizing...")
            raise ValueError("Percentiles not in [0,1] range", array.min(), array.max())
        
        assert list(array.shape) == [RES, RES, 3], (
            f"Unexpected shape: {array.shape} != {[RES, RES, 3]}"
            )

        array = np.uint8(array * 255)
        
        first_channel = array[..., 0]
        colored_array = apply_colormap(first_channel)  # Try different colormaps!
        colored_img = Image.fromarray(colored_array)
        output_path = os.path.join(percdir, f"seed{i}.png")
        colored_img.save(output_path)
        samples.append(array / 255)

    fake_u = np.array(samples)
    print("Saved all percentiles for u10 field.")

    # load training samples
    print("\nLoading training samples...")
    data = xr.open_dataset(os.path.join(datadir, 'data.nc'))
    params = data['params'].values
    train_x = data.anomaly.values
    params = np.flip(params, axis=0)
    train_x = np.flip(train_x, axis=1)
    train_x.shape
    del data

    # resize generated data to match training data
    if list(fake_u.shape[1:]) != list(train_x.shape[1:]):
        print("Generated images shape:", fake_u.shape)
        print("Training data shape:", train_x.shape)
        print("Resizing generated images...") 
        fake_u = torch.tensor(fake_u).permute(0, 3, 1, 2)
        fake_u = resize(fake_u, (TRAINRES, TRAINRES))
        fake_u = fake_u.permute(0, 2, 3, 1).numpy()
        print("Resized to:", fake_u.shape)
    else:
        print("Not resizing images.") 

    # obtain original scale data
    print("Applying GPD quantile functions...")
    fake_u[fake_u >= 1.] = (1 - 1e-6) * fake_u[fake_u >= 1.] # rescale to [0,1)
    fake = invPIT(fake_u, train_x, params)
    
    def minmax_scaler(array:np.ndarray, axis=(0,1)) -> np.ndarray:
        # (0, 1) minimum for each field
        minima = np.min(array, axis=axis, keepdims=True)
        maxima = np.max(array, axis=axis, keepdims=True)
        print(f"Size of reduced arrays: {np.min(array, axis=axis).shape}")
        return (array - minima) / (maxima - minima)
        
    # save single channel images
    print("Saving single channel images...")
    fake = minmax_scaler(fake, axis=(1, 2)) # to normalise all the same
    for i in range(len(fake)):
        array = fake[i]
        array = np.uint8(array * 255)
        # array = minmax_scaler(array, axis=(0, 1)) # to normalise per-sample
        first_channel = array[..., 0]
        colored_array = apply_colormap(first_channel)
        colored_img = Image.fromarray(colored_array)
        output_path = os.path.join(quantdir, f"seed{i}.png")
        colored_img.save(output_path)

    # make grids
    print("Creating grids...")
    perc_paths = sorted(glob.glob(os.path.join(percdir, "seed*.png")))
    quant_paths = sorted(glob.glob(os.path.join(quantdir, "seed*.png")))
    perc_paths = [path for path in perc_paths if "grid" not in path]
    quant_paths = [path for path in quant_paths if "grid" not in path]

    if len(perc_paths) > 0:
        create_image_grid(perc_paths, (8, 8), os.path.join(griddir, f"p{RES}x{RES}.png"))
    if len(quant_paths) > 0:
        create_image_grid(quant_paths, (8, 8), os.path.join(griddir, f"q{RES}x{RES}.png"))
    
    # finish up
    print("Done!")
    # %%
