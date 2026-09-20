#!/bin/bash
DATASET="TUT2017"
METHOD=$1

if [ "$METHOD" != "zeroshot" ] && [ "$METHOD" != "coop" ] && [ "$METHOD" != "cocoop" ] ; then
    echo "Invalid METHOD=$METHOD . Please choose one of the following: ['zeroshot', 'coop', 'cocoop']"
    exit 1
fi

# Check if --use_zebra is passed as an argument
USE_ZEBRA=""
for arg in "$@"; do
    if [ "$arg" = "--use_zebra" ]; then
        USE_ZEBRA="--use_zebra"
        break
    fi
done

echo "Running METHOD=$METHOD on DATASET=$DATASET"
if [ -n "$USE_ZEBRA" ]; then
    echo "Using --use_zebra flag"
else
    echo "Not using --use_zebra flag"
fi



DATA_STORAGE_DIR="/home/admin/asif.hanif"
DATASET_ROOT="$DATA_STORAGE_DIR/datasets/Audio-Datasets/$DATASET"

if [ -d "$DATASET_ROOT" ]; then
    echo "Dataset path exists: $DATASET_ROOT"
else
    echo "Dataset path does not exist. Please set the correct path to the dataset root directory in variable DATASET_ROOT"
fi


if [ "$METHOD" = "coop" ] || [ "$METHOD" = "cocoop" ]; then
    CTX_DIM=512
else
    CTX_DIM=1024
fi


if [ "$METHOD" = "zeroshot" ]; then
    SEEDS=0
else
    # SEEDS="0 1 2"
    SEEDS="0"
fi


for SEED in $SEEDS
    do
        echo "Running with SEED=$SEED"
        if [ -f "$DATASET_ROOT/train.csv" ]; then rm -rf "$DATASET_ROOT/train.csv"; fi # remove train.csv if it exists
        if [ -f "$DATASET_ROOT/test.csv" ]; then rm -rf "$DATASET_ROOT/test.csv"; fi # remove test.csv if it exists
        cp "$DATASET_ROOT/csv_files/train.csv" "$DATASET_ROOT/train.csv" # copy train.csv from csv_files to dataset root
        cp "$DATASET_ROOT/csv_files/test.csv" "$DATASET_ROOT/test.csv" # copy test.csv from csv_files to dataset root
        
        python main.py \
            --method_name $METHOD \
            $USE_ZEBRA \
            --dataset_root $DATASET_ROOT \
            --n_epochs 50 \
            --freq_test_model 10 \
            --ctx_dim $CTX_DIM \
            --batch_size 16 \
            --lr 0.05 \
            --seed $SEED \
            --exp_name "$DATASET" \
            --eval_dataset "$DATASET" \
            --num_shots 16 \
            --save_model \
            --save_model_path "$DATA_STORAGE_DIR/models/zebra" \
            --do_logging 
    done