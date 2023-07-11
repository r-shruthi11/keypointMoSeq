#!/bin/bash
#SBATCH --job-name=fit5
#SBATCH --time=71:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --chdir='/tigress/MMURTHY/usingla/keypointMoSeq'
#SBATCH --output='/scratch/gpfs/us3519/fit_pair/logs/fit.%A.%a.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

VIDEO_DIR="/tigress/MMURTHY/usingla/sampledata5/"
echo "$VIDEO_DIR"

PROJECT_DIR="/scratch/gpfs/us3519/fit_pair/project/"
echo "$PROJECT_DIR"

USE_INSTANCE=1 # 0 for female, 1 for male
echo "$USE_INSTANCE"

PY_SCRIPT_FIT="/projects/MMURTHY/usingla/keypointMoSeq/keypoint_moseq/run/fit_new_model.py"


module load cudnn/cuda-11.x/8.2.0
conda activate /tigress/MMURTHY/usingla/envs/kpms_sleap
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python $PY_SCRIPT_FIT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE
