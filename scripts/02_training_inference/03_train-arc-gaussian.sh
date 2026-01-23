#!/bin/bash
#SBATCH --job-name=gauss
#SBATCH --output=./gauss.out
#SBATCH --error=./gauss.err
#SBATCH --partition="short,interactive,medium"
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spet5107@ox.ac.uk
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

DATA="/data/ouce-opsis/spet5107/data/zipfiles/gaussian_minmax09.zip"
OUTDIR="/data/ouce-opsis/spet5107/data"
SCRIPT="/data/ouce-opsis/spet5107/hazGAN/styleGAN-DA/src/train.py"

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data/ouce-opsis/spet5107/hazGAN2/.snakemake/conda/55ad1cb60ae140a2919a9f3f8906a963_ # styleGAN snakemake env

source /data/ouce-opsis/spet5107/hazGAN2/workflow/scripts/cuda_env.sh

mkdir -p $OUTDIR

python $SCRIPT \
    --outdir=$OUTDIR \
    --data=$DATA \
    --gpus=1 \
    --DiffAugment="color,translation,cutout" \
    --kimg=300

# find $OUTDIR -name "network-snapshot-*.pkl" -not -name "network-snapshot-00300.pkl" -delete
