#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/AugValBench_v1"
PY="$HOME/miniconda3/envs/rec/bin/python"

cd "$ROOT"

export HF_HOME="/data/users/zhjia/cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

# 镜像 HTTP 下载，不走 Xet / 官方 CAS 下载链路
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
unset HF_XET_HIGH_PERFORMANCE
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=30

echo "===== HF MIRROR PIPELINE ====="
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
date

# ------------------------------------------------------------
# 网络预检：先使用当前代理，失败则尝试不使用代理
# ------------------------------------------------------------

MIRROR_TEST="https://hf-mirror.com/microsoft/Phi-3.5-mini-instruct/resolve/main/config.json"

check_mirror() {
    curl -fsSL \
        --connect-timeout 10 \
        --max-time 30 \
        -o /dev/null \
        "$MIRROR_TEST"
}

if ! check_mirror; then
    echo "Existing proxy/network path failed."
    echo "Trying direct connection without proxy..."

    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
    unset http_proxy https_proxy all_proxy

    if ! check_mirror; then
        echo "FATAL: Mirror unreachable with or without proxy."
        exit 2
    fi
fi

echo "MIRROR CONNECTION PASS"

# ------------------------------------------------------------
# 磁盘保护
# ------------------------------------------------------------

check_disk() {
    FREE_KB=$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')
    FREE_GB=$((FREE_KB / 1024 / 1024))

    echo "DISK FREE: ${FREE_GB} GiB"

    if [ "$FREE_GB" -lt 25 ]; then
        echo "FATAL: less than 25 GiB free."
        exit 3
    fi
}

# ------------------------------------------------------------
# 自动断点续传 + 有限重试
# ------------------------------------------------------------

download_model() {
    MODEL="$1"

    echo
    echo "================================================="
    echo "DOWNLOAD: $MODEL"
    echo "================================================="

    check_disk

    for ATTEMPT in $(seq 1 12); do

        echo
        echo "ATTEMPT $ATTEMPT/12: $MODEL"
        date

        if "$PY" -u - "$MODEL" <<'PY'
import sys
import json
from pathlib import Path
from huggingface_hub import snapshot_download

model = sys.argv[1]

path = Path(snapshot_download(
    repo_id=model,
    max_workers=4,
    allow_patterns=[
        "*.json",
        "*.py",
        "*.txt",
        "*.model",
        "*.jinja",
        "tokenizer*",
        "model-*.safetensors",
        "model.safetensors",
    ],
))

# 检查权重文件不是仅有配置/分词器
weights = (
    list(path.glob("model-*.safetensors"))
    + list(path.glob("model.safetensors"))
)

if not weights:
    raise RuntimeError(
        f"No model weight files found: {path}"
    )

index = path / "model.safetensors.index.json"

if index.exists():
    meta = json.loads(index.read_text())
    required = set(meta["weight_map"].values())
    missing = [
        name for name in required
        if not (path / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Missing weight shards: {missing}"
        )

for weight in weights:
    if weight.stat().st_size == 0:
        raise RuntimeError(
            f"Empty weight file: {weight}"
        )

print("DOWNLOAD VERIFIED:", model)
print("SNAPSHOT:", path)
print("WEIGHT FILES:", len(weights))
print(
    "WEIGHT GiB:",
    round(
        sum(p.stat().st_size for p in weights)
        / (1024 ** 3),
        2,
    ),
)
PY
        then
            echo "DOWNLOAD SUCCESS: $MODEL"
            return 0
        fi

        echo "Download failed; retry in 60 seconds..."
        sleep 60
    done

    echo "FATAL DOWNLOAD FAILURE: $MODEL"
    return 1
}

download_model "microsoft/Phi-3.5-mini-instruct"
download_model "mistralai/Mistral-7B-Instruct-v0.3"

echo
echo "========================================"
echo "BOTH MODELS DOWNLOADED"
echo "========================================"

# ------------------------------------------------------------
# 下载完成才启动实验
# ------------------------------------------------------------

FREE_KB=$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))

if [ "$FREE_GB" -lt 12 ]; then
    echo "FATAL: less than 12 GiB free before experiments."
    exit 4
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo
echo "========================================"
echo "START CROSS-FAMILY EXPERIMENTS"
echo "========================================"
date

bash scripts/run_cross_family_benchmark.sh

echo
echo "========================================"
echo "ALL EXPERIMENTS FINISHED"
echo "========================================"
date
