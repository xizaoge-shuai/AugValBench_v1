#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/AugValBench_v1"
PY="$HOME/miniconda3/envs/rec/bin/python"

cd "$ROOT"

# ============================================================
# HuggingFace / Xet
# ============================================================

export HF_HOME="/data/users/zhjia/cache/huggingface"

# 当前 huggingface_hub 已弃用 hf_transfer。
unset HF_HUB_ENABLE_HF_TRANSFER || true

# 真正启用当前版本的高性能 Xet 下载。
export HF_XET_HIGH_PERFORMANCE=1

# 镜像现在只有 KB/s，而且通常不能发挥 Xet。
# 下载阶段改回 HuggingFace 官方端点。
unset HF_ENDPOINT || true

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

MODELS=(
    "microsoft/Phi-3.5-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)


download_model () {

    MODEL="$1"

    echo
    echo "================================================================"
    echo "DOWNLOAD MODEL"
    echo "model = $MODEL"
    echo "HF_HOME = $HF_HOME"
    echo "HF_XET_HIGH_PERFORMANCE = $HF_XET_HIGH_PERFORMANCE"
    echo "================================================================"

    # snapshot_download 会自动复用 cache 和 .incomplete 文件。
    # 外层 retry 是为了服务器/网络临时断开后自动继续。
    for ATTEMPT in $(seq 1 20)
    do

        echo
        echo "------------------------------------------------------------"
        echo "DOWNLOAD ATTEMPT $ATTEMPT/20"
        echo "$MODEL"
        echo "------------------------------------------------------------"

        if "$PY" -u - "$MODEL" <<'PY'
import sys

from huggingface_hub import (
    snapshot_download,
)

model = sys.argv[1]

print(
    "snapshot_download:",
    model,
    flush=True,
)

path = snapshot_download(
    repo_id=model,
    max_workers=16,
)

print(
    "DOWNLOAD COMPLETE:",
    model,
    flush=True,
)

print(
    "snapshot:",
    path,
    flush=True,
)
PY
        then

            echo
            echo "SUCCESS: $MODEL"
            return 0

        fi

        echo
        echo "DOWNLOAD FAILED TEMPORARILY:"
        echo "$MODEL"
        echo "sleep 30 seconds, then resume..."

        sleep 30

    done

    echo
    echo "FATAL: failed after 20 attempts:"
    echo "$MODEL"

    return 1
}


# ============================================================
# Download both model families.
# ============================================================

for MODEL in "${MODELS[@]}"
do
    download_model "$MODEL"
done


echo
echo "================================================================"
echo "ALL MODEL DOWNLOADS COMPLETE"
echo "================================================================"


# ============================================================
# Cache verification
# ============================================================

echo
echo "===== MODEL CACHE SIZE ====="

du -sh \
"$HF_HOME/hub/models--microsoft--Phi-3.5-mini-instruct" \
2>/dev/null || true

du -sh \
"$HF_HOME/hub/models--mistralai--Mistral-7B-Instruct-v0.3" \
2>/dev/null || true


echo
echo "===== REMAINING INCOMPLETE FILES ====="

find \
"$HF_HOME/hub/models--microsoft--Phi-3.5-mini-instruct" \
"$HF_HOME/hub/models--mistralai--Mistral-7B-Instruct-v0.3" \
-type f \
-name "*.incomplete" \
-print \
2>/dev/null || true


# ============================================================
# Important:
# Experiment must now use ONLY local model files.
#
# This prevents benchmark execution from silently blocking
# because of HuggingFace network requests.
# ============================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo
echo "================================================================"
echo "START CROSS-FAMILY BENCHMARK"
echo "================================================================"

bash \
scripts/run_cross_family_benchmark.sh


echo
echo "================================================================"
echo "DOWNLOAD + CROSS-FAMILY PIPELINE DONE"
echo "================================================================"
