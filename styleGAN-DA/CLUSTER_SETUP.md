The following steps set up the environment correctly on the SoGE cluster:
```bash
micromamba create -n "styleGAN" python=3.7
micromamba activate styleGAN

python -m pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -c "import torch; print(torch.__version__)" # 1.7.1+cu110
 
python -m pip install psutil scipy click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
 
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd stylegan2-ada-pytorch

# run on GPU
srun -p GPU --pty /bin/bash

nvidia-smi      # output below
nvcc --version  # output below

python train.py --outdir=training-runs --data=100-shot-pandas.zip --gpus=1 --dry-run
python train.py --outdir=training-runs --data=100-shot-pandas.zip --gpus=1
```
However, the SoGE cluster has multiple versions of CUDA, which you can check in `ls /usr/local/`. We want plain old `cuda` which corresponds to CUDA 11.x. Make sure your `.bashrc` points to the correct files by inserting the following lines
```bash
# ===============================
# CUDA Configuration
# ===============================

# ls /usr/local/
CUDA_VERSION="cuda"
# CUDA_VERSION="cuda-12.6"
export CUDA_PATH="/usr/local/${CUDA_VERSION}"
export CUDA_ROOT="${CUDA_PATH}"
export CUDA_HOME="${CUDA_PATH}"
export CUDA_HOST_COMPILER="/usr/bin/"
export PATH="${CUDA_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"

```



## Checks

```bash
$ nvidia-smi
Mon Jan 13 12:23:00 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1080 Ti     Off |   00000000:02:00.0 Off |                  N/A |
|  0%   18C    P8              7W /  250W |       2MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce GTX 1080 Ti     Off |   00000000:82:00.0 Off |                  N/A |
|  0%   19C    P8              7W /  250W |       2MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce GTX 1080 Ti     Off |   00000000:83:00.0 Off |                  N/A |
|  0%   16C    P8              7W /  250W |       2MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
$ /usr/local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

---
Other package configurations Eman tried that didn't work (CUDA 11.0):
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
wheel
python -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```