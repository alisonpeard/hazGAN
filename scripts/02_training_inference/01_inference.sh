#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inference.out
#SBATCH --error=inference.err
#SBATCH --partition=GPU
#SBATCH --time=05:00:00
#SBATCH --dependency=afterok:116190

MODEL="00024"
STEP=300
DATADIR=../../../hazGAN-data/stylegan_output/${MODEL}

source /lustre/soge1/users/spet5107/micromamba/etc/profile.d/micromamba.sh

micromamba activate styleGAN
python ../../src/generate.py --outdir=${DATADIR}/results/trunc-1_0 --seeds=1-5000 --trunc=1.0 --network=${DATADIR}/network-snapshot-$(printf "%06d" $STEP).pkl

micromamba activate hazGANv0
python inference.py --model=${MODEL} --step=${STEP} 
