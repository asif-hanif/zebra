#!/bin/bash
bash scripts/testing/run_all_datasets_zeroshot.sh base
bash scripts/testing/run_all_datasets_zeroshot.sh novel
bash scripts/testing/run_all_datasets_coop.sh base 
bash scripts/testing/run_all_datasets_coop.sh base --use_zebra   # COOP with ZEBRA
bash scripts/testing/run_all_datasets_coop.sh novel
bash scripts/testing/run_all_datasets_coop.sh novel --use_zebra   # COOP with ZEBRA
bash scripts/testing/run_all_datasets_cocoop.sh base
bash scripts/testing/run_all_datasets_cocoop.sh base --use_zebra   # COCOOP with ZEBRA
bash scripts/testing/run_all_datasets_cocoop.sh novel
bash scripts/testing/run_all_datasets_cocoop.sh novel --use_zebra   # COCOOP with ZEBRA

