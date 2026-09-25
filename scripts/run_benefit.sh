#!/usr/bin/env bash

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rec

cd ~/AugValBench_v1

echo "================================================"
echo "STEP 1: Build benefit dataset"
echo "================================================"
python -m src.benefit.build_dataset

echo
echo "================================================"
echo "STEP 2: Train OOF benefit predictors"
echo "================================================"
python -m src.benefit.train_predictor

echo
echo "================================================"
echo "STEP 3: Evaluate benefit routers"
echo "================================================"
python -m src.benefit.evaluate_router

echo
echo "================================================"
echo "DONE"
echo "================================================"
