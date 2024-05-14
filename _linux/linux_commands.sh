# start micromamba in linux
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc
micromamba activate tensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/micromamba/envs/tensorflow/lib