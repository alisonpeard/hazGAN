#!/bin/bash
#SBATCH --job-name=arrayTest
#SBATCH --output=arrayTest_%A_%a.out
#SBATCH --array=1-4
#SBATCH --time=00:05:00

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID