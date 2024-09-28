#!/bin/bash
set -eo pipefail

NUM_BATCHES=100
NUM_EPOCHS=100
export NUM_BATCHES
export NUM_EPOCHS

./bonus_task/run.sh
