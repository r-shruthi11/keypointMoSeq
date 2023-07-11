# Usage: bash fit_hyperparams.sh

JOB_SCRIPT="/projects/MMURTHY/usingla/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_hyperparams_array.sh"
NAME="2023_07_06-14_18_16"
ARRAY_ARGS_FILE="/scratch/gpfs/us3519/fit_pair/project/${NAME}/array_args.txt" # Path to the directory containing the nested hyperparams directory
VID_DIR="/tigress/MMURTHY/usingla/sampledata/from_shruthi_scratch/data/"
USE_INSTANCE=1

echo $ARRAY_ARGS_FILE

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Submitting $NUM_ARRAY_JOBS jobs"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$VID_DIR" $USE_INSTANCE
