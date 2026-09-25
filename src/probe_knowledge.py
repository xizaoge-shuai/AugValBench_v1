"""
Symmetric / position-debiased collaborative knowledge probing.

For each (anchor, positive, negative), evaluate two orders:

Forward:
    Option1 = positive
    Option2 = negative

Swapped:
    Option1 = negative
    Option2 = positive

Let:
    m_f = logit(1) - logit(2) on Forward
    m_s = logit(1) - logit(2) on Swapped

Then:
    semantic_margin = (m_f - m_s) / 2
    position_bias   = (m_f + m_s) / 2

knowledge_score = sigmoid(semantic_margin)
"""

import json
import math
import pandas as pd
from tqdm import tqdm

from src import config
from src.llm_backend import BinaryLLMScorer
from src.utils import read_jsonl


def title(meta, item_id):
    return meta.get(item_id, {}).get("title", item_id)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def make_prompt(scorer, anchor_title, option1_title, option2_title):
    text = f"""
We want to compare two candidate products relative to an anchor product.

Anchor product:
{anchor_title}

Option 1:
{option1_title}

Option 2:
{option2_title}

Which option is MORE likely to have a collaborative relationship with the anchor product?

Answer with exactly one digit:
1 = Option 1
2 = Option 2
""".strip()
    return scorer.chat_prompt(text)


def main():
    probes = read_jsonl(config.DATA_DIR / "knowledge_probes.jsonl")

    with (config.DATA_DIR / "item_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    scorer = BinaryLLMScorer()

    forward_prompts = []
    swapped_prompts = []

    for p in tqdm(probes, desc="Build symmetric knowledge prompts"):
        anchor = title(meta, p["item_id"])
        pos = title(meta, p["positive_neighbor"])
        neg = title(meta, p["negative_neighbor"])

        forward_prompts.append(make_prompt(scorer, anchor, pos, neg))
        swapped_prompts.append(make_prompt(scorer, anchor, neg, pos))

    print("Scoring forward order ...")
    margin_forward = scorer.option1_minus_option2_margin(forward_prompts)

    print("Scoring swapped order ...")
    margin_swapped = scorer.option1_minus_option2_margin(swapped_prompts)

    rows = []
    for p, mf, ms in zip(probes, margin_forward, margin_swapped):
        semantic_margin = (mf - ms) / 2.0
        position_bias = (mf + ms) / 2.0
        knowledge_score = sigmoid(semantic_margin)

        rows.append({
            **p,
            "margin_forward": mf,
            "margin_swapped": ms,
            "semantic_margin": semantic_margin,
            "position_bias_margin": position_bias,
            "knowledge_score": knowledge_score,
            "correct_pred": int(knowledge_score > 0.5),
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No knowledge probes survived v2 filtering.")

    raw.to_csv(config.RESULT_DIR / "knowledge_probe_raw.csv", index=False)

    agg = raw.groupby("item_id").agg(
        knowledge_score=("knowledge_score", "mean"),
        probe_accuracy=("correct_pred", "mean"),
        n_probes=("probe_id", "count"),
        mean_co_count=("co_count", "mean"),
        mean_association_score=("association_score", "mean"),
        mean_semantic_margin=("semantic_margin", "mean"),
        mean_position_bias=("position_bias_margin", "mean"),
        mean_abs_position_bias=("position_bias_margin", lambda s: s.abs().mean()),
    ).reset_index()

    agg.to_csv(config.RESULT_DIR / "item_knowledge.csv", index=False)

    print("\n===== GLOBAL PROBE DIAGNOSTIC =====")
    print(f"Probe count: {len(raw)}")
    print(f"Debiased probe accuracy: {raw['correct_pred'].mean():.4f}")
    print(f"Mean knowledge score: {raw['knowledge_score'].mean():.4f}")
    print(f"Mean semantic margin: {raw['semantic_margin'].mean():.4f}")
    print(f"Mean position-bias margin: {raw['position_bias_margin'].mean():.4f}")
    print(f"Mean |position-bias margin|: {raw['position_bias_margin'].abs().mean():.4f}")

    print("\n===== ITEM-LEVEL SUMMARY =====")
    print(agg.describe(include="all").to_string())
    print("\nSaved:", config.RESULT_DIR / "item_knowledge.csv")


if __name__ == "__main__":
    main()
