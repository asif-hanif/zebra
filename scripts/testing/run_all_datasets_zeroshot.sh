#!/bin/bash

EVAL_TYPE=${1:-"novel"}  # Default to "novel" if not provided

# Base or Novel Validation
if [ "$EVAL_TYPE" != "base" ] && [ "$EVAL_TYPE" != "novel" ] && [ "$EVAL_TYPE" != "all" ]; then
    echo "Invalid EVAL_TYPE=$EVAL_TYPE. Please choose 'base', 'novel', or 'all'"
    exit 1
fi

bash scripts/testing/beijing_opera.sh zeroshot $EVAL_TYPE
bash scripts/testing/crema_d.sh zeroshot $EVAL_TYPE
bash scripts/testing/esc50_actions.sh zeroshot $EVAL_TYPE
bash scripts/testing/esc50.sh zeroshot $EVAL_TYPE
bash scripts/testing/gt_music_genre.sh zeroshot $EVAL_TYPE
bash scripts/testing/ns_instruments.sh zeroshot $EVAL_TYPE
bash scripts/testing/ravdess.sh zeroshot $EVAL_TYPE
bash scripts/testing/sesa.sh zeroshot $EVAL_TYPE
bash scripts/testing/tut.sh zeroshot $EVAL_TYPE
bash scripts/testing/urban_sound.sh zeroshot $EVAL_TYPE
bash scripts/testing/vocal_sound.sh zeroshot $EVAL_TYPE
