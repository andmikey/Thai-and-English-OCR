#!/bin/bash
set -eo pipefail

python3 ../assignment_code/train_model.py --data ~/assignment_1/runs/test_run/data --batches 64 --epochs 100 --save-dir ~/assignment_1/runs/test_run/output