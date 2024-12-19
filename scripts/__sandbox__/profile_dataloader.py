"""
Find the memory leak!


https://www.omi.me/blogs/tensorflow-guides/how-to-fix-tensorflow-memory-leaks
"""
# %%
global FORCE_CPU

RUN_EAGERLY = False
RESTRICT_MEMORY = True
MEMORY_LIMIT = 612
MEMORY_GROWTH = False
LOG_DEVICE_PLACEMENT = False
VERBOSE = True
FORCE_CPU= False


import os
import sys
import yaml
from environs import Env
import tensorflow as tf
from tensorflow import keras
import hazGAN as hazzy

import gc
import tracemalloc
from memory_profiler import profile, memory_usage

tf.keras.backend.clear_session()
tf.debugging.set_log_device_placement(LOG_DEVICE_PLACEMENT)
tf.config.run_functions_eagerly(RUN_EAGERLY) # for debugging


plot_kwargs = {"bbox_inches": "tight", "dpi": 300}


global run
global datadir
global rundir
global imdir
global runname


def config_tf_devices():
    """Use GPU if available and set memory configuration."""
    gpus = tf.config.list_physical_devices("GPU")
    if (not gpus) or FORCE_CPU:
        cpus = tf.config.list_logical_devices("CPU")
        device = cpus[0].device_type
        print(f"Using CPU: {device}")
    else:
        try:
            if RESTRICT_MEMORY:
                print(f"Restricting memory to {MEMORY_LIMIT} bytes")
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=MEMORY_LIMIT)]
                    )
            elif MEMORY_GROWTH:
                print(f"Setting memory growth to {MEMORY_GROWTH}")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, MEMORY_GROWTH)
        except Exception as e:
            raise e
        
        device = gpus[0].device_type
        print(f"Using GPU: {device}")
    return device

@profile(stream=open('dataloader.log', 'w+'))
def iterate(dataloader, epochs):
    i = 0
    tracemalloc.start()
    for batch in dataloader:
        uniform = batch['uniform']
        label = batch['label']
        condition = batch['condition']
        i += 1

        gc.collect()
        keras.backend.clear_session()
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        
        if i >= epochs:
            break

if __name__ == "__main__":
    device = config_tf_devices()

    # define paths
    env = Env()
    env.read_env(recurse=True)
    workdir = env.str("WORKINGDIR")
    datadir = env.str('TRAINDIR')
    imdir = os.path.join(workdir, "figures", "temp")

    config_tf_devices()

    with open(os.path.join(os.path.dirname(__file__), "config-defaults.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)
    config = {key: value['value'] for key, value in config.items()}
    run = None

    with tf.device(device): # <--- This is the culprit
        train, valid, metadata = hazzy.load_data(datadir)
        train = train.prefetch(tf.data.AUTOTUNE)
        valid = valid.prefetch(tf.data.AUTOTUNE)

        print("Data sizes:\n-----------")
        print("train:", sys.getsizeof(train), ' bytes')
        print("valid:", sys.getsizeof(valid), ' bytes')
        print("metadata:", sys.getsizeof(metadata), ' bytes')

        iterate(train, 100)
    
# %%