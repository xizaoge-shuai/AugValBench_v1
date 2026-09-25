import json
import pandas as pd
from tqdm import tqdm

import config
from llm_backend import BinaryLLMScorer
from utils import read_jsonl


def title(meta, item_id):
    return meta.get(item_id, {}).get("title", item_id)


def main():
    probes = read_jsonl(config.DATA_DIR / "knowledge_probes.jsonl")

    with (config.DATA_DIR / "item_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    scorer = BinaryLLMScorer()

    prompts = []
    correct = []

    for p in tqdm(probes, desc="Build knowledge prompts"):
        text = f"""
We want to test whether two products are likely to be interacted with by similar users.

Anchor product:
{title(meta, p["item_id"])}

Option 1:
{title(meta, p["option1"])}

Option 2:
{title(meta, p["option2"])}

Which option is MORE likely to have a collaborative relationship with the anchor product?

Answer with exactly one digit:
1 = Option 1
2 = Option 2
""".strip()

        prompts.append(scorer.chat_prompt(text))
        correct.append(int(p["correct_option"]))

    probs = scorer.prob_correct_option(prompts, correct)

    rows = []
    for p, prob in zip(probes, probs):
        rows.append({
            **p,
            "correct_probability": prob,
            "correct_pred": int(prob >= 0.5),
        })

    raw = pd.DataFrame(rows)
    raw.to_csv(config.RESULT_DIR / "knowledge_probe_raw.csv", index=False)

    agg = raw.groupby("item_id").agg(
        knowledge_score=("correct_probability", "mean"),
        probe_accuracy=("correct_pred", "mean"),
        n_probes=("probe_id", "count"),
        mean_co_count=("co_count", "mean"),
    ).reset_index()

    agg.to_csv(config.RESULT_DIR / "item_knowledge.csv", index=False)

    print(agg.describe(include="all"))
    print("Saved:", config.RESULT_DIR / "item_knowledge.csv")


if __name__ == "__main__":
    main()
