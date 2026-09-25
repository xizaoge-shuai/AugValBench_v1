import json
import math
import random
from pathlib import Path
from typing import Iterable, Dict, Any, List
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dcg_single_positive(rank_1based: int, k: int = 10) -> float:
    if rank_1based <= 0 or rank_1based > k:
        return 0.0
    return 1.0 / math.log2(rank_1based + 1.0)


def hit_single_positive(rank_1based: int, k: int = 10) -> float:
    return float(0 < rank_1based <= k)


def mrr_single_positive(rank_1based: int) -> float:
    return 0.0 if rank_1based <= 0 else 1.0 / rank_1based


def safe_text(x, max_chars: int = 500) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        x = " ".join(str(v) for v in x if v)
    if isinstance(x, dict):
        x = " ".join(f"{k}: {v}" for k, v in list(x.items())[:8])
    x = " ".join(str(x).split())
    return x[:max_chars]
