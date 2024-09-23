#!/bin/bash
set -eo pipefail

experiment_path=$(pwd)
# Set up directories
mkdir -p $experiment_path/data/ $experiment_path/outputs/

# Generate training data
python3 ../../assignment_code/bonus_task.py -c "200dpi_BW" -t Book
