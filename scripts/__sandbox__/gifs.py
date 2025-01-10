# %% https://wandb.ai/_scott/gif-maker/reports/Create-Gifs-From-Images-Logged-to-Weights-Biases--VmlldzoyMTI4NDQx
from environs import Env
env = Env()
env.read_env(recurse=True)
imdir = env.str("IMAGEDIR")
print(imdir)
# %%
import wandb
api = wandb.Api()
run = api.run("alison-peard/hazGAN-linux/jnymqsd6")
# %%
import os
os.chdir(imdir)
os.makedirs("gifs", exist_ok=True)
os.chdir("gifs")
# %%
for file in run.files():
    if file.name.endswith('.png'):
        file.download()

# %%
from PIL import Image

DURATION = 200 # milliseconds

def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.name.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, duration=DURATION, loop=0)

# %%
    w, h = frames[0].size
    cmd = f"ffmpeg -loglevel error -i {f'{fname}.gif'} -vcodec libx264 -crf 25 -pix_fmt yuv420p {f'{fname}.mp4'}"
    os.system(cmd)
    if not os.path.exists(f'{fname}.mp4'):
        print(f"Failed to create mp4 file.")

# %%