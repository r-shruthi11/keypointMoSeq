#!/bin/bash

video_dir="/tigress/MMURTHY/usingla/sampledata/from_shruthi_scratch/data"
save_dir="/scratch/gpfs/us3519/fit_pair/project"

run_script="generate_hyperparams_configs.py"

conda activate /tigress/MMURTHY/usingla/envs/kpms_sleap

python "$run_script" -v "$video_dir" -s "$save_dir"