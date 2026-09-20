#!/bin/bash
# Check if --use_zebra is passed as an argument

USE_ZEBRA=""
for arg in "$@"; do
    if [ "$arg" = "--use_zebra" ]; then
        USE_ZEBRA="--use_zebra"
        break
    fi
done

bash scripts/training/beijing_opera.sh coop $USE_ZEBRA
bash scripts/training/crema_d.sh coop $USE_ZEBRA
bash scripts/training/esc50_actions.sh coop $USE_ZEBRA
bash scripts/training/esc50.sh coop $USE_ZEBRA
bash scripts/training/gt_music_genre.sh coop $USE_ZEBRA
bash scripts/training/ns_instruments.sh coop $USE_ZEBRA
bash scripts/training/ravdess.sh coop $USE_ZEBRA
bash scripts/training/sesa.sh coop $USE_ZEBRA
bash scripts/training/tut.sh coop $USE_ZEBRA
bash scripts/training/urban_sound.sh coop $USE_ZEBRA
bash scripts/training/vocal_sound.sh coop $USE_ZEBRA