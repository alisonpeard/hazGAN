"""Load the generated samples and training data and plot metrics.
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

THRESHOLD = [None, 15.][1]
MONTH     = 9
NYEARS    = 500
DOMAINS    = ["uniform", "gaussian", "gumbel", "rescaled"]
VERSION    = ["", "-04", "-05", "-06"][0] # for different experiments with same domain

def load_pngs(png_dir):
    png_list = os.listdir(png_dir)
    png_list = [png for png in png_list if not png.startswith(".")]
    png_list = [os.path.join(png_dir, png) for png in png_list]
    png_list = sorted(png_list)

    samples = []
    for png in (pbar := tqdm(png_list, desc=f"Loading files from {png_dir}")):
        pbar.set_description(f"Loading {os.path.basename(png)}")
        with Image.open(png) as img:
            samples.append(np.array(img))
    samples = np.array(samples).astype(float)
    samples /= 255.
    print(f"Loaded {samples.shape} samples")
    return samples

FIELD = 0
# %%
# %%
thresh = 0.8
for DOMAIN in ["gaussian"]:
    TRAIN     = f"/soge-home/projects/mistral/alison/hazGAN-data/training/images/{DOMAIN}/storm"
    SAMPLES   = f"/soge-home/projects/mistral/alison/hazGAN-data/stylegan_output/{DOMAIN}{VERSION}/gen"
    train = load_pngs(TRAIN)
    # gen   = load_pngs(SAMPLES)
    # headroom = gen.max() - train.max()
    # print(f"Headroom between training and generated samples for {DOMAIN}: {headroom}")
    
    train = train[..., FIELD].ravel()
    train = train[train > thresh]

    # gen = gen[..., FIELD].ravel()
    # gen = gen[gen > thresh]

    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kws = dict(density=True, bins=50, alpha=0.5, edgecolor='k', linewidth=0.1)
    ax.hist(train, label="Training data", **hist_kws);
    # ax.hist(gen, label="Generated samples", **hist_kws);
    ax.legend(loc="upper right")
    fig.suptitle(f"Field {FIELD} using {DOMAIN}")
    ax.set_xlim([thresh, 1.0])
    plt.show()
    # print(sum(gen == 1.) / len(gen) * 100., "% of generated samples at one")

# %%
thresh = 0.8
for DOMAIN in DOMAINS:
    TRAIN     = f"/soge-home/projects/mistral/alison/hazGAN-data/training/images/{DOMAIN}/storm"
    SAMPLES   = f"/soge-home/projects/mistral/alison/hazGAN-data/stylegan_output/{DOMAIN}{VERSION}/gen"
    train = load_pngs(TRAIN)
    gen   = load_pngs(SAMPLES)
    headroom = gen.max() - train.max()
    print(f"Headroom between training and generated samples for {DOMAIN}: {headroom}")
    
    train = train[..., FIELD].ravel()
    train = train[train > thresh]

    gen = gen[..., FIELD].ravel()
    gen = gen[gen > thresh]

    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kws = dict(density=True, bins=50, alpha=0.5, edgecolor='k', linewidth=0.1)
    ax.hist(train, label="Training data", **hist_kws);
    ax.hist(gen, label="Generated samples", **hist_kws);
    ax.legend(loc="upper right")
    fig.suptitle(f"Field {FIELD} using {DOMAIN}")
    plt.show()
    print(sum(gen == 1.) / len(gen) * 100., "% of generated samples at one")


# %%
