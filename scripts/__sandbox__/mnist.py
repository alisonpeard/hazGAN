# %%
PLOT = True
DATAROOT = "/Users/alison/Documents/DPhil/github/stylgan/input/"
IMAGE_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 1
SAMPLES_PER_EPOCH = 100

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
os.environ["KERAS_BACKEND"] = "torch"
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from hazGAN.torch import WGANGP # noqa: E402

NUM_WORKERS = os.cpu_count() - 1

class CustomDatasetWrapper(Dataset):
    """Make standard PyTorch datasets compatible with the my training loop"""
    def __init__(self, original_dataset):
        self.n = len(original_dataset)
        labels = torch.randint(0, 3, (self.n,))
        conditions = torch.randint(0, 3, (self.n,))
        self.data= {
            "uniform": original_dataset,
            "label": labels,
            "condition": conditions
        }

    def __getitem__(self, index):
        return {
            "uniform": self.data["uniform"][index][0],
            "label": self.data["label"][index],
            "condition": self.data["condition"][index]
        }

    def __len__(self):
        return self.n

if __name__ == "__main__":
    # script begins
    print("Loading dataset...")
    dataset = datasets.ImageFolder(root=DATAROOT,
                                transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataset2 = CustomDatasetWrapper(dataset)
    # %%
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
    start = time.time()
    print("Attempting iteration...")
    iterator = iter(dataloader)
    print("Iterator created")

    print("Attempting to get first batch...")
    batch = next(iterator)
    print("First batch loaded!")
    print(f"Time taken for standard dataset: {time.time() - start:.2f} seconds")

    #%%
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS)
    start = time.time()
    print("Attempting iteration...")
    iterator = iter(dataloader2)
    print("Iterator created")

    print("Attempting to get first batch...")
    batch = next(iterator)
    print("First batch loaded!")
    print(f"Time taken for custom dataset: {time.time() - start:.2f} seconds")

    # %%

                                            
    # %%
    print(f"Loaded {len(dataset):,.0f} images")

    device = "mps" if torch.mps.is_available() else "cpu"

    if PLOT:
        print("Plotting a batch of images...")
        real_batch = next(iter(dataloader))['uniform']
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
                (1,2,0)
            ))
        plt.show()

    print("Compiling model...")
    model = WGANGP(device=device)
    model.compile()

    print("Beginning training...")
    history = model.fit(dataloader, epochs=EPOCHS,
                        steps_per_epoch=(SAMPLES_PER_EPOCH // BATCH_SIZE)
                        )

# %%