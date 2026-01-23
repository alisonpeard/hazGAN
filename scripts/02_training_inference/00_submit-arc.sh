#!/bin/bash

# just run on headnode with bash, i.e. "bash 00_submit-arc.sh"
SCALINGS=("rp10000")
DOMAINS=("gaussian") # "gumbel" "uniform") # "rescaled"
FORMATS=("npy") # ("npy" "png")
KIMG=10

for SCALING in "${SCALINGS[@]}"; do
  for DOMAIN in "${DOMAINS[@]}"; do
    for FORMAT in "${FORMATS[@]}"; do
      sbatch --export=SCALING=$SCALING,DOMAIN=$DOMAIN,FORMAT=$FORMAT,KIMG=$KIMG \
        --job-name=${DOMAIN} \
        --output=./${SCALING}_${DOMAIN}_${FORMAT}.out \
        --error=./${SCALING}_${DOMAIN}_${FORMAT}.err \
        00_train-arc.sh
    done
  done
done
