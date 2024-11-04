# For hazGAN [Tensorflow]
## Check cluster nodes status
```bash
sinfo
srun -w ouce-cn19 --pty /bin/bash
```

## Set up hazGAN environment from YAML
```bash
cd hazGAN/environments
micromamba create -f linux-gpu.yaml
micromamba activate hazGAN
```
If you are using GPU with CUDA. First, basic test that the GPU is working:

```bash
srun -w ouce-cn19 --pty /bin/bash
micromamba activate hazGAN
cd ../scripts/cluster

bash verify-gpu.sh

python helloworld_tensorflow.py --device CPU
python helloworld_tensorflow.py --device GPU
```

Sometimes you need to set up your environment to make sure your script can find `cudnn` (go over all this to make sure not out of date):

```bash
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs # nofix

cd ~/micromamba/envs/hazGAN-GPU/lib/python3.12/site-packages/tensorrt_libs

ln -s libnvinfer.so.10 libnvinfer.so.8.6.1l
ln -s py libnvinfer_plugin.so.8.6.1

export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/hazGAN-GPU/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/hazGAN-GPU/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}
```

I still see the following warnings whenever I initialize a tensorflow script. When working on a shared server with multiple CUDA versions, these warnings are common and usually not critical.

```bash
2024-11-04 20:14:45.534997: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1730751285.555829  295528 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1730751285.562363  295528 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-11-04 20:14:45.586065: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
```
Claude-suggested workarounds:
```bash
# Set CUDA 12.4 as it matches your nvcc version
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
This is just because there are multiple versions of CUDA installed on the server.

# For StylGAN2+DiffAugment
## Check CUDA version
```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

## Set up environment
Install Pytorch 3.7:
```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ micromamba install -c conda-forge pytorch=1.7.1
conda-forge/noarch                                  16.7MB @   8.6MB/s  2.0s
conda-forge/linux-64                                38.7MB @  11.5MB/s  3.4s

Pinned packages:
  - python 3.7.*


Transaction

  Prefix: /lustre/soge1/users/spet5107/micromamba/envs/DiffAug

  Updating specs:

   - pytorch=1.7.1


  Package              Version  Build               Channel           Size
────────────────────────────────────────────────────────────────────────────
  Install:
────────────────────────────────────────────────────────────────────────────

  + python_abi             3.7  2_cp37m             conda-forge     Cached
  + llvm-openmp         19.1.0  h84d6215_0          conda-forge     Cached
  + future              0.18.2  py37h89c1867_5      conda-forge     Cached
  + libgcc              14.1.0  h77fa898_1          conda-forge     Cached
  + mkl                 2020.4  h726a3e6_304        conda-forge     Cached
  + libblas              3.9.0  8_mkl               conda-forge     Cached
  + ninja               1.11.0  h924138e_0          conda-forge     Cached
  + liblapack            3.9.0  8_mkl               conda-forge     Cached
  + libcblas             3.9.0  8_mkl               conda-forge     Cached
  + numpy               1.21.6  py37h976b520_0      conda-forge     Cached
  + typing_extensions    4.7.1  pyha770c72_0        conda-forge     Cached
  + pycparser             2.21  pyhd8ed1ab_0        conda-forge     Cached
  + cffi                1.15.1  py37h43b0acd_1      conda-forge     Cached
  + pytorch              1.7.1  cpu_py37hf1c21f6_2  conda-forge     Cached

  Change:
────────────────────────────────────────────────────────────────────────────

  - _libgcc_mutex          0.1  main                pkgs/main       Cached
  + _libgcc_mutex          0.1  conda_forge         conda-forge     Cached

  Upgrade:
────────────────────────────────────────────────────────────────────────────

  - libgomp             11.2.0  h1234567_1          pkgs/main       Cached
  + libgomp             14.1.0  h77fa898_1          conda-forge     Cached
  - libgcc-ng           11.2.0  h1234567_1          pkgs/main       Cached
  + libgcc-ng           14.1.0  h69a702a_1          conda-forge     Cached

  Downgrade:
────────────────────────────────────────────────────────────────────────────

  - _openmp_mutex          5.1  1_gnu               pkgs/main       Cached
  + _openmp_mutex          4.5  2_kmp_llvm          conda-forge     Cached

  Summary:

  Install: 14 packages
  Change: 1 packages
  Upgrade: 2 packages
  Downgrade: 1 packages

  Total download: 0 B

────────────────────────────────────────────────────────────────────────────


Confirm changes: [Y/n] y

Transaction starting
Changing _libgcc_mutex-0.1-main ==> _libgcc_mutex-0.1-conda_forge
Linking python_abi-3.7-2_cp37m
Linking llvm-openmp-19.1.0-h84d6215_0
Changing libgomp-11.2.0-h1234567_1 ==> libgomp-14.1.0-h77fa898_1
Linking future-0.18.2-py37h89c1867_5
Changing _openmp_mutex-5.1-1_gnu ==> _openmp_mutex-4.5-2_kmp_llvm
Linking libgcc-14.1.0-h77fa898_1
warning  libmamba [libgcc-14.1.0-h77fa898_1] The following files were already present in the environment:
    - lib/libatomic.so
    - lib/libatomic.so.1
    - lib/libatomic.so.1.2.0
    - lib/libgcc_s.so
    - lib/libgcc_s.so.1
    - lib/libitm.so
    - lib/libitm.so.1
    - lib/libitm.so.1.0.0
    - lib/libquadmath.so
    - lib/libquadmath.so.0
    - lib/libquadmath.so.0.0.0
    - share/info/libgomp.info
    - share/info/libquadmath.info
    - share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION
Linking mkl-2020.4-h726a3e6_304
Changing libgcc-ng-11.2.0-h1234567_1 ==> libgcc-ng-14.1.0-h69a702a_1
Linking libblas-3.9.0-8_mkl
Linking ninja-1.11.0-h924138e_0
Linking liblapack-3.9.0-8_mkl
Linking libcblas-3.9.0-8_mkl
Linking numpy-1.21.6-py37h976b520_0
Linking typing_extensions-4.7.1-pyha770c72_0
Linking pycparser-2.21-pyhd8ed1ab_0
Linking cffi-1.15.1-py37h43b0acd_1
Linking pytorch-1.7.1-cpu_py37hf1c21f6_2

Transaction finished

To activate this environment, use:

    micromamba activate DiffAug

Or to execute a single command in this environment, use:

    micromamba run -n DiffAug mycommand
```

Install extra packages:
```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ python -m pip install click requests tqdm pyspng ninja psutil Pillow scipy
Collecting click
  Using cached click-8.1.7-py3-none-any.whl (97 kB)
Collecting requests
  Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Collecting tqdm
  Using cached tqdm-4.66.5-py3-none-any.whl (78 kB)
Collecting pyspng
  Using cached pyspng-0.1.2-cp37-cp37m-linux_x86_64.whl
Collecting ninja
  Using cached ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
Collecting psutil
  Using cached psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (290 kB)
Collecting Pillow
  Using cached Pillow-9.5.0-cp37-cp37m-manylinux_2_28_x86_64.whl (3.4 MB)
Collecting scipy
  Using cached scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (38.1 MB)
Collecting importlib-metadata
  Using cached importlib_metadata-6.7.0-py3-none-any.whl (22 kB)
Requirement already satisfied: certifi>=2017.4.17 in /lustre/soge1/users/spet5107/micromamba/envs/DiffAug/lib/python3.7/site-packages (from requests) (2022.12.7)
Collecting urllib3<3,>=1.21.1
  Using cached urllib3-2.0.7-py3-none-any.whl (124 kB)
Collecting idna<4,>=2.5
  Using cached idna-3.10-py3-none-any.whl (70 kB)
Collecting charset-normalizer<4,>=2
  Using cached charset_normalizer-3.3.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (136 kB)
Requirement already satisfied: numpy in /lustre/soge1/users/spet5107/micromamba/envs/DiffAug/lib/python3.7/site-packages (from pyspng) (1.21.6)
Collecting zipp>=0.5
  Using cached zipp-3.15.0-py3-none-any.whl (6.8 kB)
Requirement already satisfied: typing-extensions>=3.6.4 in /lustre/soge1/users/spet5107/micromamba/envs/DiffAug/lib/python3.7/site-packages (from importlib-metadata->click) (4.7.1)
Installing collected packages: ninja, zipp, urllib3, tqdm, scipy, pyspng, psutil, Pillow, idna, charset-normalizer, requests, importlib-metadata, click
Successfully installed Pillow-9.5.0 charset-normalizer-3.3.2 click-8.1.7 idna-3.10 importlib-metadata-6.7.0 ninja-1.11.1.1 psutil-6.0.0 pyspng-0.1.2 requests-2.31.0 scipy-1.7.3 tqdm-4.66.5 urllib3-2.0.7 zipp-3.15.0
```

```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ python -c "import torch;print(torch.cuda.is_available())"
False
```
### Checking everything again
```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ which python
/lustre/soge1/users/spet5107/micromamba/envs/DiffAug/bin/python
```

```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

```
(DiffAug) spet5107@ouce-cn24 /soge-home/projects/mistral/alison/data-efficient-gans/DiffAugment-stylegan2-pytorch (ecdf-only)
$ 
```