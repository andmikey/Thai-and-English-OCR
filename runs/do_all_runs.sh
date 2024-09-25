#!/bin/bash
set -eo pipefail

NUM_BATCHES=20
NUM_EPOCHS=50
export NUM_BATCHES
export NUM_EPOCHS

# Activate conda env
conda activate /scratch/gusandmich/assignment_1_scratch

run_names=('all_thai_all_thai' 'all_thai_english_jointly'
    'thai_bold_thai_normal' 'thai_english_normal_jointly'
    'thai_normal_200' 'thai_normal_400_thai_bold_400'
    'thai_normal_400_thai_normal_200')

# run_names=('thai_normal_200') # For testing things work

for run_name in "${run_names[@]}"; do
    bash $run_name/run.sh
done
