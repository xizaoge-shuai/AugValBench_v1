import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUGVAL_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
RESULT_DIR = Path(os.environ.get("AUGVAL_RESULT_DIR", str(ROOT / "results"))).expanduser().resolve()

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR = RESULT_DIR / "plots"

for p in [DATA_DIR, RESULT_DIR, PLOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

AMAZON_CATEGORY = os.environ.get("AUGVAL_AMAZON_CATEGORY", "All_Beauty")
DATASET_NAME = os.environ.get("AUGVAL_DATASET_NAME", AMAZON_CATEGORY)
POSITIVE_RATING = 4.0
MIN_USER_INTERACTIONS = 3
MIN_ITEM_INTERACTIONS = 3
MAX_KCORE_ROUNDS = 10
MAX_EVAL_USERS = 1000000
NEGATIVE_COUNT = 19
MAX_HISTORY = 10
RANDOM_SEED = 2027

MODEL_NAME = os.environ.get("AUGVAL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOAD_IN_4BIT = True
BATCH_SIZE = 8
MAX_PROMPT_TOKENS = 1024

CONDITIONS = [
    "base",
    "clean",
]

# Knowledge probing v2
KNOWLEDGE_PROBES_PER_ITEM = 5
MIN_COOCCURRENCE = 2
HARD_NEGATIVE_TOPK = 30
PROBE_ONLY_EVAL_ITEMS = True

DEBUG = False
DEBUG_USERS = 20
