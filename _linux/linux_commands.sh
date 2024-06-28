# $HOME
/soge-home/users/spet5107

# check gpu
nvidia-smi
nvcc --version

# check my quotas
bash /ouce-home/projects/mistral/bin/quotas

# example .bashrc file from Tom
source/less mistral/docs/example.bashrc

# create a symlink
ln -s \path\to\file \alias\

# search for a string recursively
grep -r "string"

# start micromamba in linux
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc
micromamba activate test-gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/micromamba/envs/tensorflow/lib

# set up to run hazGAN model
exec bash
micromamba env create -n hazGAN -f _/linux/hazGAN.yaml
export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/hazGAN/lib
touch train.sh
echo "#!/bin/bash -x" >> train.sh
echo "which python" >> train.sh
echo "train.py --dry-run" >> train.sh
chmod +x train.sh
srun -p GPU --gres=gpu:tesla:1 --pty dry-run.sh

srun -p GPU --gres=gpu:tesla:1 --pty nvidia-smi

# interactive shell with GPU
srun -p GPU --gres=gpu:tesla:1 --pty /bin/bash

# activate GPU environment
micromamba activate hazGAN-GPU
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs
cd ~/micromamba/envs/hazGAN-GPU/lib/python3.12/site-packages/tensorrt_libs/
cd ~/micromamba/envs/gpu-test/lib/python3.12/site-packages/tensorrt_libs
ln -s libnvinfer.so.10 libnvinfer.so.8.6.1l
ln -s py libnvinfer_plugin.so.8.6.1
export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/gpu-test/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/soge-home/users/spet5107/micromamba/envs/gpu-test/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

# Errror
nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 550.90

# Train single script
python train.py --device cpu --cluster --dry-run