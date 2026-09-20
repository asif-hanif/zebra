#!/bin/bash
bash scripts/training/run_all_datasets_coop.sh 
bash scripts/training/run_all_datasets_coop.sh --use_zebra   # COOP with ZEBRA
bash scripts/training/run_all_datasets_cocoop.sh 
bash scripts/training/run_all_datasets_cocoop.sh --use_zebra   # COCOOP with ZEBRA

