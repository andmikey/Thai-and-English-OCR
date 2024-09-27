#!/bin/bash
set -eo pipefail

NUM_BATCHES=10
NUM_EPOCHS=20
export NUM_BATCHES
export NUM_EPOCHS

run_names=('all_thai' 'all_thai_english_jointly'
    'thai_bold_thai_normal' 'thai_english_normal_jointly'
    'thai_normal_200' 'thai_normal_400_thai_bold_400'
    'thai_normal_400_thai_normal_200')

for run_name in "${run_names[@]}"; do
    ./$run_name/run.sh
done
