#!/bin/bash

# Request interaactive node - do this on the CLI
# salloc --nodes=1 --ntasks=1 --mem=4G --time=00:30:00 --gres=gpu:1

# Activate conda environment
module load anaconda3/2023.3
conda activate /tigress/MMURTHY/usingla/envs/kpms_sleap

# Test arguments
SAVE_DIR="/scratch/gpfs/shruthi/pair_wt_gold/fitting/"
VID_DIR="/scratch/gpfs/shruthi/pair_wt_gold/"
# PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_to_test_hyperparams.py"
PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/generate_hyperparams_configs.py"


# Run script
python "$PY_SCRIPT" -s "$SAVE_DIR" -v "$VID_DIR"
