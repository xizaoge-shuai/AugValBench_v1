import json
import random

import numpy as np
import pandas as pd

from src import config
from src.utils import dcg_single_positive, hit_single_positive, mrr_single_positive


def load_popularity():
    with (config.DATA_DIR / "item_popularity.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def evaluate_case(df):
    x = df.sort_values("chosen_score", ascending=False).reset_index(drop=True)
    pos = x.index[x["is_target"] == 1].tolist()
    if not pos:
        return None

    rank = pos[0] + 1
    return {
        "rank": rank,
        "ndcg10": dcg_single_positive(rank, 10),
        "hr10": hit_single_positive(rank, 10),
        "mrr": mrr_single_positive(rank),
    }


def main():
    scores = pd.read_csv(config.RESULT_DIR / "rerank_scores.csv")
    popularity = load_popularity()

    kb_path = config.RESULT_DIR / "item_knowledge.csv"
    if kb_path.exists():
        kb = pd.read_csv(kb_path)[["item_id", "knowledge_score"]]
        kb["item_id"] = kb["item_id"].astype(str)
    else:
        kb = pd.DataFrame(columns=["item_id", "knowledge_score"])

    base = scores[scores["condition"] == "base"][
        ["case_id", "user_id", "item_id", "is_target", "score"]
    ].rename(columns={"score": "base_score"})

    clean = scores[scores["condition"] == "clean"][
        ["case_id", "item_id", "score"]
    ].rename(columns={"score": "clean_score"})

    z = base.merge(clean, on=["case_id", "item_id"], how="inner")
    z["item_id"] = z["item_id"].astype(str)
    z = z.merge(kb, on="item_id", how="left")
    z["popularity"] = z["item_id"].map(popularity).fillna(0.0)

    knowledge_threshold = z["knowledge_score"].median()
    popularity_threshold = z["popularity"].median()

    rng = random.Random(config.RANDOM_SEED)
    rows = []

    for case_id, g in z.groupby("case_id"):
        variants = {}

        x = g.copy()
        x["chosen_score"] = x["base_score"]
        variants["NoAugment"] = x

        x = g.copy()
        x["chosen_score"] = x["clean_score"]
        variants["UniformClean"] = x

        x = g.copy()
        mask = np.array([rng.random() < 0.5 for _ in range(len(x))])
        x["chosen_score"] = np.where(mask, x["clean_score"], x["base_score"])
        variants["Random50"] = x

        x = g.copy()
        mask = x["popularity"] <= popularity_threshold
        x["chosen_score"] = np.where(mask, x["clean_score"], x["base_score"])
        variants["TailSelective"] = x

        x = g.copy()
        mask = x["knowledge_score"].fillna(1.0) <= knowledge_threshold
        x["chosen_score"] = np.where(mask, x["clean_score"], x["base_score"])
        variants["LowKnowledgeSelective"] = x

        x = g.copy()
        x["chosen_score"] = np.where(
            x["is_target"] == 1,
            np.maximum(x["base_score"], x["clean_score"]),
            np.minimum(x["base_score"], x["clean_score"]),
        )
        variants["OracleUpperBound"] = x

        for name, x in variants.items():
            m = evaluate_case(x)
            if m is None:
                continue
            rows.append({
                "case_id": case_id,
                "baseline": name,
                **m,
            })

    out = pd.DataFrame(rows)
    out.to_csv(config.RESULT_DIR / "baseline_metrics.csv", index=False)

    summary = out.groupby("baseline")[["ndcg10", "hr10", "mrr", "rank"]].mean()
    summary.to_csv(config.RESULT_DIR / "baseline_summary.csv")

    print(summary)
    print("\nknowledge threshold =", knowledge_threshold)
    print("popularity threshold =", popularity_threshold)


if __name__ == "__main__":
    main()
