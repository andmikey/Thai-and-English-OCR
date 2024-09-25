#!/bin/bash
set -eo pipefail

experiment_path=$(pwd)
# Set up directories
mkdir -p $experiment_path/data/ $experiment_path/outputs/

# Generate training data with default ratio of train/test/val (0.6/0.2/0.2)
python3 ../../assignment_code/generate_training_data.py \
    -l Thai -d 200 -s normal \
    --output_path $experiment_path/data/

# Train the model
python3 ../../assignment_code/train_model.py \
    --train-data $experiment_path/data/training_set.txt \
    --validation-data $experiment_path/data/validation_set.txt \
    --save-dir $experiment_path/outputs/ \
    --batches $NUM_BATCHES --epochs $NUM_EPOCHS

# Evaluate the model
python3 ../../assignment_code/evaluate_model.py \
    --test-data $experiment_path/data/testing_set.txt \
    --model-path $experiment_path/outputs/ \
    --save-dir $experiment_path/outputs/
