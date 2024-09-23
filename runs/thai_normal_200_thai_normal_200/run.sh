#!/bin/bash
set -eo pipefail

experiment_path=$(pwd)
# Set up directories
mkdir -p $experiment_path/data/ $experiment_path/outputs/

# Generate training data
python3 ../../assignment_code/generate_training_data.py \
    -l Thai -d 200 -s normal \
    --output_path $experiment_path/data/

# Train the model
python3 ../../assignment_code/train_model.py \
    --train-data $experiment_path/data/training_set.txt \
    --validation-data $experiment_path/data/validation_set.txt \
    --save-dir $experiment_path/outputs/ \
    --batches 1 --epochs 1

# Evaluate the model
python3 ../../assignment_code/evaluate_model.py \
    --data $experiment_path/data/testing.txt \
    --model-path $experiment_path/outputs/ \
    --save-dir $experiment_path/outputs/
