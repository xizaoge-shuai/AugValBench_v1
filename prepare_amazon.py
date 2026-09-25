from collections import Counter, defaultdict
import json
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import config
from utils import set_seed, safe_text, write_jsonl


def iterative_kcore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _ in range(config.MAX_KCORE_ROUNDS):
        before = len(out)
        user_cnt = out["user_id"].value_counts()
        item_cnt = out["parent_asin"].value_counts()

        valid_users = user_cnt[user_cnt >= config.MIN_USER_INTERACTIONS].index
        valid_items = item_cnt[item_cnt >= config.MIN_ITEM_INTERACTIONS].index

        out = out[
            out["user_id"].isin(valid_users)
            & out["parent_asin"].isin(valid_items)
        ].copy()

        if len(out) == before:
            break
    return out


def normalize_meta(row):
    return {
        "item_id": str(row.get("parent_asin", "")),
        "title": safe_text(row.get("title", ""), 240),
        "main_category": safe_text(row.get("main_category", ""), 120),
        "categories": safe_text(row.get("categories", []), 240),
        "features": safe_text(row.get("features", []), 500),
        "description": safe_text(row.get("description", []), 700),
        "store": safe_text(row.get("store", ""), 120),
        "details": safe_text(row.get("details", {}), 500),
    }


def main():
    set_seed(config.RANDOM_SEED)
    rng = random.Random(config.RANDOM_SEED)

    print(f"[1/5] Loading raw_review_{config.AMAZON_CATEGORY} ...")
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{config.AMAZON_CATEGORY}",
        split="full",
        trust_remote_code=True,
    )

    cols = ["user_id", "parent_asin", "rating", "timestamp"]
    df = reviews.select_columns(cols).to_pandas()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=cols)
    df = df[df["rating"] >= config.POSITIVE_RATING].copy()
    df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")

    print("before k-core:", len(df))
    df = iterative_kcore(df)
    print("after  k-core:", len(df))

    if df.empty:
        raise RuntimeError("k-core 后为空，请降低 config.py 中的 k-core 阈值。")

    df = df.sort_values(["user_id", "timestamp"])

    user_all_items = defaultdict(set)
    train_rows = []
    raw_cases = []

    for uid, g in df.groupby("user_id", sort=False):
        seq = g["parent_asin"].astype(str).tolist()
        if len(seq) < 3:
            continue

        target = seq[-1]
        history = seq[:-1]

        user_all_items[str(uid)] = set(seq)
        train_rows.append({"user_id": str(uid), "items": history})
        raw_cases.append({
            "user_id": str(uid),
            "history": history,
            "target": target,
        })

    pop = Counter()
    for r in train_rows:
        pop.update(r["items"])

    item_universe = sorted(pop.keys())

    rng.shuffle(raw_cases)
    max_users = config.DEBUG_USERS if config.DEBUG else config.MAX_EVAL_USERS
    raw_cases = raw_cases[:max_users]

    cases = []
    for idx, r in enumerate(raw_cases):
        forbidden = user_all_items[r["user_id"]]
        neg_pool = [
            x for x in item_universe
            if x not in forbidden and x != r["target"]
        ]
        if len(neg_pool) < config.NEGATIVE_COUNT:
            continue

        negatives = rng.sample(neg_pool, config.NEGATIVE_COUNT)
        candidates = negatives + [r["target"]]
        rng.shuffle(candidates)

        cases.append({
            "case_id": idx,
            "user_id": r["user_id"],
            "history": r["history"][-config.MAX_HISTORY:],
            "target": r["target"],
            "candidates": candidates,
        })

    write_jsonl(config.DATA_DIR / "train_sequences.jsonl", train_rows)
    write_jsonl(config.DATA_DIR / "eval_cases.jsonl", cases)

    with (config.DATA_DIR / "item_popularity.json").open("w", encoding="utf-8") as f:
        json.dump(pop, f, ensure_ascii=False)

    print(f"[3/5] Loading raw_meta_{config.AMAZON_CATEGORY} ...")
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{config.AMAZON_CATEGORY}",
        split="full",
        trust_remote_code=True,
    )

    needed = set(item_universe)
    for c in cases:
        needed.add(c["target"])
        needed.update(c["history"])
        needed.update(c["candidates"])

    meta = {}
    for row in tqdm(meta_ds, total=len(meta_ds)):
        item_id = str(row.get("parent_asin", ""))
        if item_id in needed:
            m = normalize_meta(row)
            if m["title"]:
                meta[item_id] = m

    with (config.DATA_DIR / "item_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    kept = []
    for c in cases:
        if c["target"] not in meta:
            continue

        history = [x for x in c["history"] if x in meta]
        candidates = [x for x in c["candidates"] if x in meta]

        if len(history) < 2 or c["target"] not in candidates or len(candidates) < 5:
            continue

        c["history"] = history
        c["candidates"] = candidates
        kept.append(c)

    write_jsonl(config.DATA_DIR / "eval_cases.jsonl", kept)

    print(f"Final eval cases: {len(kept)}")
    print(f"Metadata items: {len(meta)}")
    print("Saved to:", config.DATA_DIR)


if __name__ == "__main__":
    main()
