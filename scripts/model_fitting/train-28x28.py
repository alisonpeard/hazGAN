
# %%
import hazGAN.keras.model as setup
# %%
"""
## Prepare the Fashion-MNIST data

To demonstrate how to train WGAN-GP, we will be using the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Each
sample in this dataset is a 28x28 grayscale image associated with a label from
10 classes (e.g. trouser, pullover, sneaker, etc.)
"""
import os
import keras
import hazGAN as hg
import tensorflow as tf

BATCH_SIZE = 64 #512
TRAIN_SIZE = 1000 # 60000
DATA = "era5"

if DATA == "fashion_mnist":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(f"Number of examples: {len(train_images)}")
    print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

    # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
    train_images = train_images.reshape(train_images.shape[0], *setup.IMG_SHAPE).astype("float32")
    train_images = (train_images - 127.5) / 127.5
    train_images = train_images[:TRAIN_SIZE, ...]
elif DATA == "era5":
    wd = os.path.join(os.getcwd(), "..", '..')  # hazGAN directory
    datadir = os.path.join(wd, "..", "era5_data.nosync", "res_28x28") 
    data = hg.load_training(datadir, TRAIN_SIZE, None, setup.IMG_SHAPE[:2], gumbel_marginals=False)
    train_u = data['train_u']#.astype("float32")
    test_u = data['test_u']#.astype("float32")
    train_images = tf.data.Dataset.from_tensor_slices(train_u).batch(BATCH_SIZE)
    test_images = tf.data.Dataset.from_tensor_slices(test_u).batch(BATCH_SIZE)
else:
    raise ValueError("Unknown data source")
# %%

# Get the wgan model
wgan = setup.WGAN(
    discriminator=setup.d_model,
    generator=setup.g_model,
    latent_dim=setup.noise_dim,
    discriminator_extra_steps=3,
)

# Compile the wgan model
wgan.compile(
    d_optimizer=setup.discriminator_optimizer,
    g_optimizer=setup.generator_optimizer,
    g_loss_fn=setup.generator_loss,
    d_loss_fn=setup.discriminator_loss,
)

# Set the number of epochs for training.
epochs = 500

# Instantiate the customer `GANMonitor` Keras callback.
cbk = setup.GANMonitor(num_img=3, latent_dim=setup.noise_dim)

# Start training
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
# %%
"""
Display the last generated images:
"""

from IPython.display import Image, display

display(Image(f"generated_img_0_{epochs-1}.png"))
display(Image(f"generated_img_1_{epochs-1}.png"))
display(Image(f"generated_img_2_{epochs-1}.png"))
# %%

# reproducibility
runname = "28x28_images"
rundir = os.path.join(wd, "_wandb-runs", runname)
os.makedirs(rundir, exist_ok=True)
wgan.generator.save_weights(os.path.join(rundir, f"generator.weights.h5"))
wgan.discriminator.save_weights(os.path.join(rundir, f"critic.weights.h5"))
#%%
random_latent_vectors = tf.random.normal(shape=(1000, 128))
samples = wgan.generator(random_latent_vectors)
# %%
import matplotlib.pyplot as plt
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(samples[i, :, :, 0], cmap="gray")
    ax.invert_yaxis()
    ax.axis("off")
# %%
# compare to training data
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(train_u[i, :, :, 0], cmap="gray")
    ax.invert_yaxis()
    ax.axis("off")
# %%
