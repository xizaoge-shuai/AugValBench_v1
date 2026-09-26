#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage:"
    echo "$0 DATASET MODEL SOURCE_DATA SOURCE_RESULT BENCH_RESULT"
    exit 1
fi

DATASET="$1"
MODEL="$2"
SOURCE_DATA="$3"
SOURCE_RESULT="$4"
BENCH_RESULT="$5"

ROOT="$HOME/AugValBench_v1"
PY="$HOME/miniconda3/envs/rec/bin/python"

BENCH_DATA="$ROOT/benchmark_runs/$DATASET/data"
PERTURB_RESULT="$BENCH_RESULT/perturb"
FINAL_RESULT="$BENCH_RESULT/results"

mkdir -p \
"$BENCH_DATA" \
"$PERTURB_RESULT" \
"$FINAL_RESULT"

cd "$ROOT"

echo "================================================"
echo "AUGVALBENCH"
echo "dataset = $DATASET"
echo "model   = $MODEL"
echo "================================================"

# ========================================================
# Test-only benchmark view
# ========================================================

cp \
"$SOURCE_DATA/formal/eval_test.jsonl" \
"$BENCH_DATA/eval_cases.jsonl"

cp \
"$SOURCE_DATA/train_sequences.jsonl" \
"$BENCH_DATA/train_sequences.jsonl"

cp \
"$SOURCE_DATA/item_popularity.json" \
"$BENCH_DATA/item_popularity.json"

cp \
"$SOURCE_DATA/item_meta.json" \
"$BENCH_DATA/item_meta.json"

N=$(wc -l < "$BENCH_DATA/eval_cases.jsonl")

echo "benchmark test cases = $N"

# ========================================================
# Only compute missing interventions.
# Base/Clean are reused from existing experiments.
# ========================================================

export AUGVAL_DATASET_NAME="$DATASET"
export AUGVAL_AMAZON_CATEGORY="$DATASET"
export AUGVAL_MODEL_NAME="$MODEL"
export AUGVAL_DATA_DIR="$BENCH_DATA"
export AUGVAL_RESULT_DIR="$PERTURB_RESULT"

export AUGVAL_CONDITIONS="conflict,irrelevant,partial,redundant"

echo
echo "===== RUN FOUR PERTURBATIONS ====="

"$PY" -m src.run_rerank

# ========================================================
# Merge existing Base/Clean + new perturbations
# ========================================================

echo
echo "===== ASSEMBLE SIX CONDITIONS ====="

"$PY" -m src.benchmark.assemble \
--source-result "$SOURCE_RESULT" \
--perturb-result "$PERTURB_RESULT" \
--benchmark-result "$FINAL_RESULT" \
--benchmark-data "$BENCH_DATA"

# ========================================================
# Benchmark analysis
# ========================================================

export AUGVAL_RESULT_DIR="$FINAL_RESULT"

echo
echo "===== ORIGINAL CONDITION ANALYSIS ====="

"$PY" -m src.analyze_gain

echo
echo "===== AUGVALBENCH ANALYSIS ====="

"$PY" -m src.benchmark.analyze_benchmark

echo
echo "================================================"
echo "BENCHMARK DONE"
echo "$DATASET × $MODEL"
echo "================================================"
