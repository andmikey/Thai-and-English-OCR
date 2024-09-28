#!/bin/bash
set -eo pipefail

NUM_BATCHES=256
NUM_EPOCHS=100

experiment_path=$(pwd)/$(dirname "$0")
echo $experiment_path

# Set up directories
mkdir -p $experiment_path/data/ $experiment_path/outputs/

# Generate training data on all letters + styles + DPIs
# Use 90% of data for train and 10% for validation
# (just to check the model is performing ok)
# python3 ../assignment_code/generate_training_data.py \
#     -l Thai -l English -l Numeric -l Special \
#     -trp 0.9 -vap 0.1 -tep 0 \
#     --output_path $experiment_path/data/ \
#     --logging_path $experiment_path/results.log

# # Train a character-level model on all the training data
# python3 ../assignment_code/train_model.py \
#     --train-data $experiment_path/data/training_set.txt \
#     --validation-data $experiment_path/data/validation_set.txt \
#     --save_dir $experiment_path/outputs/ \
#     --batches $NUM_BATCHES --epochs $NUM_EPOCHS \
#     --logging_path $experiment_path/results.log

# Extract + predict letters
python3 ../assignment_code/bonus_task.py -d '200' -t Book --model_path $experiment_path/outputs/model.pth --write_images
