#!/bin/bash

EVAL_TYPE=${1:-"novel"}  # Default to "novel" if not provided

# Base or Novel Validation
if [ "$EVAL_TYPE" != "base" ] && [ "$EVAL_TYPE" != "novel" ] && [ "$EVAL_TYPE" != "all" ]; then
    echo "Invalid EVAL_TYPE=$EVAL_TYPE. Please choose 'base', 'novel', or 'all'"
    exit 1
fi

USE_ZEBRA=""
for arg in "$@"; do
    if [ "$arg" = "--use_zebra" ]; then
        USE_ZEBRA="--use_zebra"
        break
    fi
done

echo "USE_ZEBRA=$USE_ZEBRA"

bash scripts/testing/beijing_opera.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/crema_d.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/esc50_actions.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/esc50.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/gt_music_genre.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/ns_instruments.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/ravdess.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/sesa.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/tut.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/urban_sound.sh coop $EVAL_TYPE $USE_ZEBRA
bash scripts/testing/vocal_sound.sh coop $EVAL_TYPE $USE_ZEBRA