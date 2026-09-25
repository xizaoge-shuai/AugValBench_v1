"""
Reranking v2.

Adds two continuous utility metrics:
- pairwise_win_rate: fraction of negatives beaten by the target
- pairwise_margin: target_score - mean(negative_score)
"""

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.interventions import augmentation_for
from src.llm_backend import BinaryLLMScorer
from src.utils import (
    read_jsonl,
    dcg_single_positive,
    hit_single_positive,
    mrr_single_positive,
    set_seed,
)


def item_title(meta, item_id):
    return meta.get(item_id, {}).get("title", item_id)


def build_prompt(scorer, history_titles, candidate_title, augmentation_text):
    history_text = "\n".join(f"- {x}" for x in history_titles)
    extra = ""
    if augmentation_text:
        extra = "\nAdditional candidate information:\n" + augmentation_text

    text = f"""
A user previously interacted with these products:
{history_text}

Candidate product:
{candidate_title}
{extra}

Based only on the user's history and the candidate information above,
is this user likely to interact with the candidate next?

Answer with exactly one digit:
1 = likely
0 = unlikely
""".strip()
    return scorer.chat_prompt(text)


def main():
    set_seed(config.RANDOM_SEED)
    cases = read_jsonl(config.DATA_DIR / "eval_cases.jsonl")

    with (config.DATA_DIR / "item_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    scorer = BinaryLLMScorer()
    score_rows = []

    for cond_idx, condition in enumerate(config.CONDITIONS):
        print("\n=== condition:", condition, "===")
        prompts = []
        prompt_meta = []

        for case in tqdm(cases, desc=f"build/{condition}"):
            history_titles = [
                item_title(meta, x)
                for x in case["history"]
                if x in meta
            ]

            for pos, item_id in enumerate(case["candidates"]):
                if item_id not in meta:
                    continue

                local_seed = (
                    config.RANDOM_SEED * 1000003
                    + int(case["case_id"]) * 997
                    + cond_idx * 97
                    + pos
                )
                rng = random.Random(local_seed)
                aug = augmentation_for(item_id, condition, meta, rng)

                prompts.append(
                    build_prompt(
                        scorer,
                        history_titles,
                        item_title(meta, item_id),
                        aug,
                    )
                )
                prompt_meta.append({
                    "case_id": case["case_id"],
                    "user_id": case["user_id"],
                    "condition": condition,
                    "item_id": item_id,
                    "is_target": int(item_id == case["target"]),
                })

        probs = scorer.prob_one_vs_zero(prompts)
        for m, p in zip(prompt_meta, probs):
            score_rows.append({**m, "score": float(p)})

        pd.DataFrame(score_rows).to_csv(
            config.RESULT_DIR / "rerank_scores.partial.csv",
            index=False,
        )

    scores = pd.DataFrame(score_rows)
    scores.to_csv(config.RESULT_DIR / "rerank_scores.csv", index=False)

    metric_rows = []

    for (case_id, condition), g in scores.groupby(["case_id", "condition"]):
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        target_rows = g[g["is_target"] == 1]
        if len(target_rows) != 1:
            continue

        target_score = float(target_rows.iloc[0]["score"])
        target_rank = int(g.index[g["is_target"] == 1][0]) + 1
        neg_scores = g.loc[g["is_target"] == 0, "score"].to_numpy(dtype=float)
        if len(neg_scores) == 0:
            continue

        wins = (target_score > neg_scores).astype(float)
        ties = (target_score == neg_scores).astype(float)
        pairwise_win_rate = float(np.mean(wins + 0.5 * ties))
        pairwise_margin = float(target_score - np.mean(neg_scores))

        metric_rows.append({
            "case_id": case_id,
            "user_id": g.iloc[0]["user_id"],
            "condition": condition,
            "target_rank": target_rank,
            "target_score": target_score,
            "ndcg10": dcg_single_positive(target_rank, 10),
            "hr10": hit_single_positive(target_rank, 10),
            "mrr": mrr_single_positive(target_rank),
            "pairwise_win_rate": pairwise_win_rate,
            "pairwise_margin": pairwise_margin,
            "n_candidates": len(g),
        })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(config.RESULT_DIR / "case_metrics.csv", index=False)

    print(
        metrics.groupby("condition")[[
            "ndcg10",
            "hr10",
            "mrr",
            "target_rank",
            "pairwise_win_rate",
            "pairwise_margin",
        ]].mean()
    )


if __name__ == "__main__":
    main()
