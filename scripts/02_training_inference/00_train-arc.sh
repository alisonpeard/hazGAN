#!/bin/bash
#SBATCH --job-name=stylegan
#SBATCH --output=stylegan.out
#SBATCH --error=stylegan.err
#SBATCH --partition="short,interactive,medium"
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spet5107@ox.ac.uk
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

set -e

DATA="/data/ouce-opsis/spet5107/data/training/${SCALING}/${DOMAIN}/${FORMAT}.zip"
OUTDIR="/data/ouce-opsis/spet5107/data/models/${SCALING}/${DOMAIN}/${FORMAT}"
SCRIPT="/data/ouce-opsis/spet5107/hazGAN/styleGAN-DA/src/train.py"

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data/ouce-opsis/spet5107/hazGAN2/.snakemake/conda/55ad1cb60ae140a2919a9f3f8906a963_ # styleGAN snakemake env

source /data/ouce-opsis/spet5107/hazGAN2/workflow/scripts/cuda_env.sh

mkdir -p $OUTDIR

: '
python $SCRIPT \
    --outdir=$OUTDIR \
    --data=$DATA \
    --gpus=1 \
    --DiffAugment="color,translation,cutout" \
    --kimg=$KIMG
'

# and generate samples
SCRIPT="/data/ouce-opsis/spet5107/hazGAN/styleGAN-DA/src/generate.py"
GENDIR="/data/ouce-opsis/spet5107/data/generated/${SCALING}/${DOMAIN}/${FORMAT}"
mkdir -p ${GENDIR}

# find the latest network snapshot
KIMG_PADDED=$(printf "%06d" $KIMG)
NETWORK=$(ls ${OUTDIR}/*/network-snapshot-$KIMG_PADDED.pkl 2>/dev/null)
if [[ -z "$NETWORK" ]]; then
    echo "ERROR: No network snapshot found in ${OUTDIR}"
    exit 1
elif [[ $(echo "$NETWORK" | wc -l) -gt 1 ]]; then
    echo "ERROR: Multiple network snapshots found:"
    echo "$NETWORK"
    exit 1
fi

# generate samples
python ${SCRIPT} \
  --outdir=${GENDIR} \
  --seeds=1-914 \
  --trunc=1.0 \
  --network=${NETWORK}

# zip samples to
ZIPDIR="/data/ouce-opsis/spet5107/data/zipfiles/${SCALING}/${DOMAIN}"
mkdir -p ${ZIPDIR}
cd ${GENDIR}
zip -r ${ZIPDIR}/${FORMAT}.zip *.${FORMAT}
