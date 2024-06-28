"""Hello World for CPU/GPU testing with tensorflow.

Source: https://github.com/ovh/ai-training-examples/blob/main/notebooks/getting-started/tensorflow/basic_cpu_benchmark.ipynb
"""
import tensorflow as tf
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="CPU", help="Device to use for computation (CPU or GPU)")
args = parser.parse_args()
device = args.device

# Get the list of all logical GPU device on your notebook
DEVICES = tf.config.list_logical_devices(device)
DEVICES_NAMES = [x.name for x in DEVICES]
DEVICES_NB = len(DEVICES)

if DEVICES_NB == 0:
    raise SystemError(f'No {device} device found')
else:
    print(f'{DEVICES_NB} {device} device(s) have been found on your notebook :')

for nb in range(DEVICES_NB):
    name = DEVICES_NAMES[nb]
    print(f'* {device} n°{nb} whose name is "{name}"')
    
print('')

def random_multiply(vector_length):
    vector_1 = tf.random.normal(vector_length)
    vector_2 = tf.random.normal(vector_length)
    return vector_1 * vector_2

def operation(vector_length):
    # If you have several GPU you can select the one to use by changing the used index of GPU_DEVICES_NAMES
    with tf.device(DEVICES_NAMES[0]):
        random_multiply(vector_length)

import timeit

# We run the op once to warm up; see: https://stackoverflow.com/a/45067900
operation([1])

for i in range(8):
    vector_length = pow(10, i)
    time = timeit.timeit(f'operation([{vector_length}])', number=20, setup="from __main__ import operation")
    print(f'Operations on vector of length {vector_length} take {time}')