#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --chdir='/tigress/MMURTHY/usingla/keypointMoSeq'
#SBATCH --output='/scratch/gpfs/us3519/fit_pair/logs/fit_apply.%j.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

VIDEO_DIR="/tigress/MMURTHY/usingla/sampledata2"
echo "$VIDEO_DIR"

PROJECT_DIR="/scratch/gpfs/us3519/fit_pair/project"
echo "$PROJECT_DIR"

USE_INSTANCE=1 # 0 for female, 1 for male
echo "$USE_INSTANCE"

PY_SCRIPT_FIT="/projects/MMURTHY/usingla/keypointMoSeq/keypoint_moseq/run/fit_kpms_to_sleap.py"
# PY_SCRIPT_INFER="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/apply_kpms_to_sleap.py"

module load anaconda3/2023.3
conda activate /tigress/MMURTHY/usingla/envs/kpms_sleap

python $PY_SCRIPT_FIT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE

# python $PY_SCRIPT_INFER --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE