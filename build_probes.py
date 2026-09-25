from collections import Counter, defaultdict
import json
import random

import config
from utils import read_jsonl, write_jsonl, set_seed


def main():
    set_seed(config.RANDOM_SEED)
    rng = random.Random(config.RANDOM_SEED)

    train = read_jsonl(config.DATA_DIR / "train_sequences.jsonl")
    cases = read_jsonl(config.DATA_DIR / "eval_cases.jsonl")

    with (config.DATA_DIR / "item_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    co = defaultdict(Counter)
    all_items = set()

    for row in train:
        items = [x for x in row["items"] if x in meta]
        unique = list(dict.fromkeys(items))
        all_items.update(unique)

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

    probes = []
    pid = 0

    for item in sorted(target_items):
        if item not in meta or item not in co:
            continue

        positives = [
            (nb, cnt)
            for nb, cnt in co[item].most_common()
            if cnt >= config.MIN_COOCCURRENCE and nb in meta
        ]

        if not positives:
            positives = [
                (nb, cnt)
                for nb, cnt in co[item].most_common(5)
                if nb in meta
            ]

        if not positives:
            continue

        positives = positives[:max(10, config.KNOWLEDGE_PROBES_PER_ITEM)]

        for _ in range(config.KNOWLEDGE_PROBES_PER_ITEM):
            pos, cnt = rng.choice(positives)

            forbidden = set(co[item].keys()) | {item, pos}
            decoys = [x for x in all_items if x not in forbidden]
            if not decoys:
                continue

            neg = rng.choice(decoys)

            if rng.random() < 0.5:
                option1, option2, correct = pos, neg, 1
            else:
                option1, option2, correct = neg, pos, 2

            probes.append({
                "probe_id": pid,
                "item_id": item,
                "positive_neighbor": pos,
                "co_count": cnt,
                "option1": option1,
                "option2": option2,
                "correct_option": correct,
            })
            pid += 1

    write_jsonl(config.DATA_DIR / "knowledge_probes.jsonl", probes)

    print(
        f"Built {len(probes)} probes for "
        f"{len(set(p['item_id'] for p in probes))} items."
    )


if __name__ == "__main__":
    main()
