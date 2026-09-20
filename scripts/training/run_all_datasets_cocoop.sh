#!/bin/bash
# Check if --use_zebra is passed as an argument

USE_ZEBRA=""
for arg in "$@"; do
    if [ "$arg" = "--use_zebra" ]; then
        USE_ZEBRA="--use_zebra"
        break
    fi
done

bash scripts/training/beijing_opera.sh cocoop $USE_ZEBRA
bash scripts/training/crema_d.sh cocoop $USE_ZEBRA
bash scripts/training/esc50_actions.sh cocoop $USE_ZEBRA
bash scripts/training/esc50.sh cocoop $USE_ZEBRA
bash scripts/training/gt_music_genre.sh cocoop $USE_ZEBRA
bash scripts/training/ns_instruments.sh cocoop $USE_ZEBRA
bash scripts/training/ravdess.sh cocoop $USE_ZEBRA
bash scripts/training/sesa.sh cocoop $USE_ZEBRA
bash scripts/training/tut.sh cocoop $USE_ZEBRA
bash scripts/training/urban_sound.sh cocoop $USE_ZEBRA
bash scripts/training/vocal_sound.sh cocoop $USE_ZEBRA
