#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rec

cd ~/AugValBench_v1

echo "===== BUDGET SWEEP ====="
python -m src.benefit.evaluate_budget_curve

echo
echo "===== FEATURE ABLATION ====="
python -m src.benefit.run_ablation

echo
echo "===== DONE ====="
