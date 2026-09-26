#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage:"
    echo "  $0 <dataset> <model> <data_dir> <result_dir>"
    exit 1
fi

DATASET="$1"
MODEL="$2"
DATA_DIR="$3"
RESULT_DIR="$4"

ROOT="$HOME/AugValBench_v1"
PY="$HOME/miniconda3/envs/rec/bin/python"

cd "$ROOT"

export AUGVAL_AMAZON_CATEGORY="$DATASET"
export AUGVAL_MODEL_NAME="$MODEL"
export AUGVAL_DATA_DIR="$DATA_DIR"
export AUGVAL_RESULT_DIR="$RESULT_DIR"

mkdir -p "$RESULT_DIR"

echo "=================================================="
echo "DATASET    = $AUGVAL_AMAZON_CATEGORY"
echo "MODEL      = $AUGVAL_MODEL_NAME"
echo "DATA_DIR   = $AUGVAL_DATA_DIR"
echo "RESULT_DIR = $AUGVAL_RESULT_DIR"
echo "=================================================="

echo "===== 1/6 BUILD PROBES ====="
"$PY" -m src.build_probes

echo "===== 2/6 PROBE KNOWLEDGE ====="
"$PY" -m src.probe_knowledge

echo "===== 3/6 BASE + CLEAN RERANK ====="
"$PY" -m src.run_rerank

echo "===== 4/6 AUGMENTATION ANALYSIS ====="
"$PY" -m src.analyze_gain

echo "===== 5/6 COUNTERFACTUAL DATASET ====="
"$PY" -m src.benefit.build_dataset

echo "===== 6/6 BASELINES ====="
"$PY" -m src.formal.evaluate_baselines

echo "=================================================="
echo "DONE"
echo "DATASET = $DATASET"
echo "MODEL   = $MODEL"
echo "=================================================="
