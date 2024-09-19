#!/bin/bash
set -eo pipefail

# Generate training data that covers all the styles, DPIs, and languages
# Use --language, --dpi, --style to choose different selections of data
# and --train_proportion, --test_proportion, --validation_proportion to adjust train/test/val balance
python3 ../assignment_code/generate_training_data.py -l English -d 200 -s normal --output_path ~/assignment_1/runs/test_run/data