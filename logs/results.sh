#!/bin/bash

#############################################################################
# Process Results Without ZEBRA

METHODS=("zeroshot" "coop" "cocoop")
EVAL_TYPES=("base" "novel")

for METHOD in "${METHODS[@]}"; do
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        python process_results.py --method "$METHOD" --eval_type "$EVAL_TYPE"
    done
done

#############################################################################
# Process Results With ZEBRA

METHODS=("coop" "cocoop")
EVAL_TYPES=("base" "novel")

for METHOD in "${METHODS[@]}"; do
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        python process_results.py --method "$METHOD" --eval_type "$EVAL_TYPE" --use_zebra
    done
done

#############################################################################
# Merge Base and Novel Results Without ZEBRA

METHODS=("zeroshot" "coop" "cocoop")
EVAL_TYPES=("base" "novel")

for METHOD in "${METHODS[@]}"; do
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        python merge_base_novel_results.py --method "$METHOD" 
    done
done

#############################################################################
# Merge Base and Novel Results With ZEBRA

METHODS=("coop" "cocoop")
EVAL_TYPES=("base" "novel")

for METHOD in "${METHODS[@]}"; do
    for EVAL_TYPE in "${EVAL_TYPES[@]}"; do
        python merge_base_novel_results.py --method "$METHOD" --use_zebra
    done
done

#############################################################################

python print_results.py --use_zebra