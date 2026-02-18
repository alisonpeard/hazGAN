#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --output=./gen.out
#SBATCH --error=./gen.err
#SBATCH --partition="short,medium,interactive"
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spet5107@ox.ac.uk
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

OUTDIR="/data/ouce-opsis/spet5107/data/generated/gumbel_minmax09"
NETWORK="/data/ouce-opsis/spet5107/data/00003-gumbel_minmax09-low_shot-kimg300-color-translation-cutout/network-snapshot-000300.pkl"
SCRIPT="/data/ouce-opsis/spet5107/hazGAN/styleGAN-DA/src/generate.py"

module load Anaconda3
conda activate /data/ouce-opsis/spet5107/hazGAN2/.snakemake/conda/55ad1cb60ae140a2919a9f3f8906a963_ # styleGAN snakemake env

mkdir -p $OUTDIR

python ${SCRIPT} \
  --outdir=${OUTDIR} \
  --seeds=1-914 \
  --trunc=1.0 \
  --network=${NETWORK}
