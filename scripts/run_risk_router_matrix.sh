#!/usr/bin/env bash

set -euo pipefail

cd "$HOME/AugValBench_v1"

PY="$HOME/miniconda3/envs/rec/bin/python"


run_one () {

    DATASET="$1"
    MODEL="$2"
    DATA_DIR="$3"
    RESULT_DIR="$4"

    echo
    echo "============================================================"
    echo "$DATASET × $MODEL"
    echo "============================================================"

    AUGVAL_DATASET_NAME="$DATASET" \
    AUGVAL_MODEL_NAME="$MODEL" \
    AUGVAL_DATA_DIR="$DATA_DIR" \
    AUGVAL_RESULT_DIR="$RESULT_DIR" \
    "$PY" \
    -m src.method.risk_utility_router
}


run_one \
All_Beauty \
Qwen/Qwen2.5-3B-Instruct \
"$HOME/AugValBench_v1/runs/All_Beauty/data" \
"$HOME/AugValBench_v1/baseline_runs/All_Beauty/Qwen2.5-3B-Instruct"


run_one \
All_Beauty \
Qwen/Qwen2.5-7B-Instruct \
"$HOME/AugValBench_v1/runs/All_Beauty/data" \
"$HOME/AugValBench_v1/runs/All_Beauty/results"


run_one \
Gift_Cards \
Qwen/Qwen2.5-3B-Instruct \
"$HOME/AugValBench_v1/runs/Gift_Cards/data" \
"$HOME/AugValBench_v1/baseline_runs/Gift_Cards/Qwen2.5-3B-Instruct"


run_one \
Gift_Cards \
Qwen/Qwen2.5-7B-Instruct \
"$HOME/AugValBench_v1/runs/Gift_Cards/data" \
"$HOME/AugValBench_v1/runs/Gift_Cards/results"


run_one \
MovieLens_1M \
Qwen/Qwen2.5-3B-Instruct \
"$HOME/AugValBench_v1/baseline_runs/MovieLens_1M/data" \
"$HOME/AugValBench_v1/baseline_runs/MovieLens_1M/Qwen2.5-3B-Instruct"


run_one \
MovieLens_1M \
Qwen/Qwen2.5-7B-Instruct \
"$HOME/AugValBench_v1/baseline_runs/MovieLens_1M/data" \
"$HOME/AugValBench_v1/baseline_runs/MovieLens_1M/Qwen2.5-7B-Instruct"


echo
echo "============================================================"
echo "RISK ROUTER MATRIX DONE"
echo "============================================================"
