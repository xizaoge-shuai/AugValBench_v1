#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/AugValBench_v1"
PY="$HOME/miniconda3/envs/rec/bin/python"

MODELS=(
    "microsoft/Phi-3.5-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

DATASETS=(
    "All_Beauty"
    "Gift_Cards"
    "MovieLens_1M"
)


source_data () {

    case "$1" in

        All_Beauty)
            echo \
            "$ROOT/runs/All_Beauty/data"
            ;;

        Gift_Cards)
            echo \
            "$ROOT/runs/Gift_Cards/data"
            ;;

        MovieLens_1M)
            echo \
            "$ROOT/baseline_runs/MovieLens_1M/data"
            ;;

        *)
            echo "unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}


slug () {

    echo "$1" \
    | sed 's#/#__#g'
}


for MODEL in "${MODELS[@]}"
do

    MODEL_SLUG="$(slug "$MODEL")"

    for DATASET in "${DATASETS[@]}"
    do

        SOURCE_DATA="$(
            source_data "$DATASET"
        )"

        BENCH_DATA="$ROOT/benchmark_runs/$DATASET/data"

        SOURCE_RESULT="$ROOT/cross_model_runs/$DATASET/$MODEL_SLUG/source"

        BENCH_RESULT="$ROOT/cross_model_runs/$DATASET/$MODEL_SLUG/benchmark"


        echo
        echo "============================================================"
        echo "CROSS-FAMILY"
        echo "dataset = $DATASET"
        echo "model   = $MODEL"
        echo "============================================================"


        mkdir -p \
        "$SOURCE_RESULT" \
        "$BENCH_RESULT"


        # -----------------------------------------------------
        # Generate BASE + CLEAN score on the formal test cases.
        # Reuses exactly the benchmark formal test split.
        # -----------------------------------------------------

        if [ ! -s \
            "$SOURCE_RESULT/rerank_scores.csv" \
        ]; then

            echo \
            ">>> generating base/clean source scores"

            PYTHONUNBUFFERED=1 \
            CUDA_VISIBLE_DEVICES=0 \
            AUGVAL_DATASET_NAME="$DATASET" \
            AUGVAL_AMAZON_CATEGORY="$DATASET" \
            AUGVAL_MODEL_NAME="$MODEL" \
            AUGVAL_DATA_DIR="$BENCH_DATA" \
            AUGVAL_RESULT_DIR="$SOURCE_RESULT" \
            AUGVAL_CONDITIONS="clean" \
            "$PY" -u \
            -m src.run_rerank

        else

            echo \
            "SKIP source rerank: already exists"

        fi


        # -----------------------------------------------------
        # Sanity check source conditions
        # -----------------------------------------------------

        "$PY" - \
        "$SOURCE_RESULT/rerank_scores.csv" <<'PY'

import sys
import pandas as pd

p = sys.argv[1]

df = pd.read_csv(p)

print(
    "source rows =",
    len(df)
)

print(
    "conditions =",
    sorted(
        df["condition"]
        .astype(str)
        .unique()
        .tolist()
    )
)
PY


        # -----------------------------------------------------
        # Existing benchmark runner computes:
        #
        # partial
        # redundant
        # irrelevant
        # conflict
        #
        # and assembles with base/clean.
        # -----------------------------------------------------

        echo \
        ">>> running six-condition benchmark"

        bash \
        "$ROOT/scripts/run_benchmark_pair.sh" \
        "$DATASET" \
        "$MODEL" \
        "$SOURCE_DATA" \
        "$SOURCE_RESULT" \
        "$BENCH_RESULT"


        echo
        echo \
        "DONE: $DATASET × $MODEL"

    done

done


echo
echo "============================================================"
echo "ALL CROSS-FAMILY BENCHMARKS DONE"
echo "============================================================"
