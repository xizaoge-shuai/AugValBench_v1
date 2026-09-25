from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
PLOT_DIR = RESULT_DIR / "plots"

for p in [DATA_DIR, RESULT_DIR, PLOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

AMAZON_CATEGORY = "All_Beauty"
POSITIVE_RATING = 4.0

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
MAX_KCORE_ROUNDS = 10

MAX_EVAL_USERS = 200
NEGATIVE_COUNT = 19
MAX_HISTORY = 10
RANDOM_SEED = 2027

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LOAD_IN_4BIT = True
BATCH_SIZE = 8
MAX_PROMPT_TOKENS = 1024

CONDITIONS = [
    "base",
    "clean",
    "redundant",
    "partial",
    "irrelevant",
    "conflict",
]

KNOWLEDGE_PROBES_PER_ITEM = 3
MIN_COOCCURRENCE = 2
PROBE_ONLY_EVAL_ITEMS = True

DEBUG = True
DEBUG_USERS = 20
