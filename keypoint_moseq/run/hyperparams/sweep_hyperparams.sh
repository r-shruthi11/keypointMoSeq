# Usage: bash sweep_hyperparams.sh

JOB_SCRIPT="/projects/MMURTHY/usingla/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_hyperparams_array.sh"
WORK_DIR="/scratch/gpfs/us3519/fit_pair/project/2023_06_23-07_46_33" # Path to the directory containing the nested hyperparams directory
VID_DIR="/tigress/MMURTHY/usingla/sampledata3"

ARRAY_ARGS_FILE="$WORK_DIR/array_args.txt"
#ARRAY_ARGS_FILE="/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_04_21-20_04_07/array_args_incomplete.txt"

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Submitting $NUM_ARRAY_JOBS jobs"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$VID_DIR"