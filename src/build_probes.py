"""
Knowledge probing v2.

Fixes two issues exposed by v1:
1) strong Option-2 position bias;
2) noisy collaborative positives from weak/accidental co-occurrence.

This file only builds semantic pairs. The actual position debiasing is done in
probe_knowledge.py by evaluating both option orders.
"""

from collections import Counter, defaultdict
import json
import math
import random

from src import config
from src.utils import read_jsonl, write_jsonl, set_seed


def main():
    set_seed(config.RANDOM_SEED)
    rng = random.Random(config.RANDOM_SEED)

    train = read_jsonl(config.DATA_DIR / "train_sequences.jsonl")
    cases = read_jsonl(config.DATA_DIR / "eval_cases.jsonl")

    with (config.DATA_DIR / "item_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # user-level item support and user-level co-occurrence
    support = Counter()
    co = defaultdict(Counter)
    all_items = set()

    for row in train:
        items = [x for x in row["items"] if x in meta]
        unique = list(dict.fromkeys(items))
        if len(unique) < 2:
            continue

        all_items.update(unique)
        for x in unique:
            support[x] += 1

        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                a, b = unique[i], unique[j]
                co[a][b] += 1
                co[b][a] += 1

    if config.PROBE_ONLY_EVAL_ITEMS:
        target_items = set()
        for c in cases:
            target_items.update(c["candidates"])
    else:
        target_items = set(all_items)

    all_items = sorted(x for x in all_items if x in meta)

    def association(a, b, cnt):
        # Confidence-weighted co-occurrence cosine.
        denom = math.sqrt(max(1, support[a]) * max(1, support[b]))
        cosine = cnt / denom
        return cosine * math.log1p(cnt)

    probes = []
    pid = 0

    for item in sorted(target_items):
        if item not in meta or item not in co or support[item] == 0:
            continue

        positives = []
        for nb, cnt in co[item].items():
            if nb not in meta:
                continue
            if cnt < config.MIN_COOCCURRENCE:
                continue
            positives.append((nb, cnt, association(item, nb, cnt)))

        if not positives:
            continue

        # Prefer stronger co-occurrence first, then association strength.
        positives.sort(key=lambda x: (x[1], x[2]), reverse=True)
        positives = positives[:config.KNOWLEDGE_PROBES_PER_ITEM]

        for pos, cnt, assoc in positives:
            forbidden = set(co[item].keys()) | {item, pos}
            candidates = [x for x in all_items if x not in forbidden]
            if not candidates:
                continue

            # Popularity-matched hard negatives.
            anchor_pop = support[item]
            candidates.sort(
                key=lambda x: abs(
                    math.log1p(support[x]) - math.log1p(anchor_pop)
                )
            )
            hard_pool = candidates[:config.HARD_NEGATIVE_TOPK]
            neg = rng.choice(hard_pool)

            probes.append({
                "probe_id": pid,
                "item_id": item,
                "positive_neighbor": pos,
                "negative_neighbor": neg,
                "co_count": cnt,
                "association_score": assoc,
                "anchor_support": support[item],
                "positive_support": support[pos],
                "negative_support": support[neg],
            })
            pid += 1

    write_jsonl(config.DATA_DIR / "knowledge_probes.jsonl", probes)

    print(
        f"Built {len(probes)} probes for "
        f"{len(set(p['item_id'] for p in probes))} items."
    )
    if probes:
        print(f"Mean co_count: {sum(p['co_count'] for p in probes) / len(probes):.3f}")
        print(
            "Mean association_score: "
            f"{sum(p['association_score'] for p in probes) / len(probes):.4f}"
        )


if __name__ == "__main__":
    main()
