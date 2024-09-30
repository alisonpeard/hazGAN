"""
https://keras.io/examples/generative/wgan_gp/
https://huggingface.co/keras-io/WGAN-GP
"""
# %%
import os
import tensorflow as tf
from tensorflow import keras
import hazGAN as hg
from wgan_demo import wgan, cbk

def config_tf_devices():
    """Use GPU if available"""
    gpus = tf.config.list_logical_devices("GPU")
    gpu_names = [x.name for x in gpus]
    if (len(gpu_names) == 0):
        print("No GPU found, using CPU")
        cpus = tf.config.list_logical_devices("CPU")
        cpu_names = [x.name for x in cpus]
        return cpu_names[0]
    else:
        print(f"Using GPU: {gpu_names[0]}")
        return gpu_names[0]

wd = os.path.join('/soge-home', 'projects', 'mistral', 'alison', 'hazGAN')
# wd = os.path.join('/Users', 'alison', 'Documents', 'DPhil', 'paper1.nosync')  # hazGAN directory
datadir = os.path.join(wd, 'training', "18x22")
figdir = os.path.join(wd, "figures", "training")

# %%
noise_dim = 128
epochs = 20
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 128

data = hg.load_training(datadir, 560, padding_mode=None,
                        gumbel_marginals=False, image_shape=(28, 28),
                        channels=['u10'])
train = data['train_u']
test = data['test_u']
print(f"Training data shape: {train.shape}")

# %%
device = config_tf_devices()
with tf.device(device):
    wgan.fit(train, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])


wgan.generator.save_weights(os.path.join(figdir, f"generator.weights.h5"))
wgan.critic.save_weights(os.path.join(figdir, f"critic.weights.h5"))
#%%
from IPython.display import Image, display

display(Image(os.path.join(figdir, "generated_img_0_19.png")))
display(Image(os.path.join(figdir, "generated_img_1_19.png")))
display(Image(os.path.join(figdir, "generated_img_2_19.png")))

# %%