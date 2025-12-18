# %%
import os
import xarray as xr


model_dir = "/soge-home/projects/mistral/alison/hazGAN-data/"
models = [
    "gaussian",
    "gaussian-04", 
    "gaussian-05",
    "gaussian-06"
]

model = next(iter(models))
model_path = os.path.join(model_dir, model + ".nc")
ds = xr.open_dataset(model_path)
# %%
