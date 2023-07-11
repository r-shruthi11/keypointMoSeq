#!/bin/bash
#SBATCH --job-name=single-split
#SBATCH --time=71:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output='/scratch/gpfs/us3519/fit_pair/logs/sweep_hyp.%A.%a.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

PY_SCRIPT="/projects/MMURTHY/usingla/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_hyperparams_sleap.py"
echo "$PY_SCRIPT"

array_args_file="$1"
video_dir="$2"
use_instance="$3"

echo "$array_args_file"
echo "$video_dir"
echo "$use_instance"

linenum=$SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_TASK_ID: $linenum"
project_dir=$(sed -n "$linenum p" $array_args_file)

echo "project_dir: $project_dir"

module load cudnn/cuda-11.x/8.2.0

conda activate /tigress/MMURTHY/usingla/envs/kpms_sleap
python "$PY_SCRIPT" -p "$project_dir" -v "$video_dir" --use_instance "$use_instance"