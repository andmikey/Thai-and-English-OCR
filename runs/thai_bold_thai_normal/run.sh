#!/bin/bash
set -eo pipefail

experiment_path=$(pwd)/$(dirname "$0")
echo $experiment_path

# Set up directories
mkdir -p $experiment_path/data/ $experiment_path/outputs/

# Training data is all Thai bold text (generates just a training_set.txt file)
python3 ../assignment_code/generate_training_data.py \
    -l Thai -d 200 -s bold -trp 0.8 -tep 0 -vap 0.2 \
    --output_path $experiment_path/data/ \
    --logging_path $experiment_path/results.log

# Testing data is all Thai normal text (generates just a testing_set.txt file)
python3 ../assignment_code/generate_training_data.py \
    -l Thai -d 200 -s normal -trp 0 -tep 1 -vap 0 \
    --output_path $experiment_path/data/ \
    --logging_path $experiment_path/results.log

# Train the model
python3 ../assignment_code/train_model.py \
    --train-data $experiment_path/data/training_set.txt \
    --validation-data $experiment_path/data/validation_set.txt \
    --save_dir $experiment_path/outputs/ \
    --batches $NUM_BATCHES --epochs $NUM_EPOCHS \
    --logging_path $experiment_path/results.log

# Evaluate the model
python3 ../assignment_code/evaluate_model.py \
    --test-data $experiment_path/data/testing_set.txt \
    --model_path $experiment_path/outputs/model.pth \
    --logging_path $experiment_path/results.log
