"""Hello World for CPU/MPS testing with PyTorch.

---- Cluster examples ----
>> python hello_world.py --device cpu
>> python hello_world.py --device mps  # For Apple Silicon
"""
import torch
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (cpu|mps|cuda)")
args = parser.parse_args()
device = args.device

# Get device information
if device.lower() == "mps":
    if not torch.backends.mps.is_available():
        raise SystemError('No MPS device found. Make sure you are using macOS 12.3+')
    DEVICES_NB = 1  # MPS typically reports as a single device
    DEVICE_NAMES = ['mps']
    print('Apple Silicon MPS device found')

if device.lower() == 'cuda':
    if not torch.cuda.is_available():
        raise SystemError("No CUDA device found.")
    DEVICE_NB = torch.cuda.device_count()
    DEVICE_NAMES = ['cuda']
    print("CUDA device found.")

else:
    DEVICES_NB = 1
    DEVICE_NAMES = ['cpu']
    print('CPU device found')

print('')

def random_multiply(vector_length):
    vector_1 = torch.randn(vector_length, device=torch.device(DEVICE_NAMES[0]))
    vector_2 = torch.randn(vector_length, device=torch.device(DEVICE_NAMES[0]))
    return vector_1 * vector_2

def operation(vector_length):
    random_multiply(vector_length)

import timeit

# Warm up run
operation([1])

for i in range(8):
    vector_length = pow(10, i)
    time = timeit.timeit(f'operation([{vector_length}])', number=20, setup="from __main__ import operation")
    print(f'Operations on vector of length {vector_length} take {time}')