#!/bin/bash
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