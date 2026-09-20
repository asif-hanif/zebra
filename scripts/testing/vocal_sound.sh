#!/bin/bash
DATASET="VocalSound"
METHOD=$1
EVAL_TYPE=${2:-"novel"}  # Default to "novel" if not provided

# Base or Novel Validation
if [ "$EVAL_TYPE" != "base" ] && [ "$EVAL_TYPE" != "novel" ] && [ "$EVAL_TYPE" != "all" ]; then
    echo "Invalid EVAL_TYPE=$EVAL_TYPE. Please choose 'base', 'novel', or 'all'"
    exit 1
fi


## Method Validation
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


echo ""
echo ""
echo "METHOD=$METHOD"
echo "USE_ZEBRA=$USE_ZEBRA"
echo "DATASET=$DATASET"
echo "EVAL_TYPE=$EVAL_TYPE"
echo ""
echo ""


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
        echo -e "Running with SEED=$SEED\n\n\n"

        if [ -n "$USE_ZEBRA" ]; then
            MODEL_PATH="$DATA_STORAGE_DIR/models/zebra/$METHOD/$DATASET-SEED$SEED-ZEBRA.pth"
        else
            MODEL_PATH="$DATA_STORAGE_DIR/models/zebra/$METHOD/$DATASET-SEED$SEED.pth"
        fi

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
            --eval_dataset $DATASET \
            --num_shots 16 \
            --eval_only \
            --eval_type ${EVAL_TYPE} \
            --load_model_abs_path $MODEL_PATH \
            --do_logging
    done