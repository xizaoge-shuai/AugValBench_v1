import json
import math
import random
from collections import defaultdict

from src import config
from src.utils import read_jsonl, write_jsonl


SEED = 2027
NEGATIVE_COUNT = 19
POOL_MULTIPLIER = 20


def main():
    rng = random.Random(SEED)

    cases = read_jsonl(
        config.DATA_DIR / "eval_cases_uniform.jsonl"
    )

    train = read_jsonl(
        config.DATA_DIR / "train_sequences.jsonl"
    )

    with (
        config.DATA_DIR / "item_popularity.json"
    ).open("r", encoding="utf-8") as f:
        popularity = {
            str(k): float(v)
            for k, v in json.load(f).items()
        }

    # 用户历史 + target，不能采成 negative
    user_seen = defaultdict(set)

    for row in train:
        user_seen[str(row["user_id"])].update(
            str(x)
            for x in row["items"]
        )

    all_items = [
        x
        for x in popularity
        if popularity.get(x, 0) > 0
    ]

    new_cases = []

    for case in cases:
        uid = str(case["user_id"])
        target = str(case["target"])

        forbidden = (
            set(user_seen[uid])
            | {target}
        )

        target_pop = popularity.get(
            target,
            0.0
        )

        target_log_pop = math.log1p(
            target_pop
        )

        candidates = []

        for item in all_items:
            if item in forbidden:
                continue

            diff = abs(
                math.log1p(
                    popularity.get(
                        item,
                        0.0
                    )
                )
                - target_log_pop
            )

            candidates.append(
                (
                    diff,
                    rng.random(),
                    item,
                )
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        # 先取与 target popularity 最近的一小片区域，
        # 再随机抽，避免 19 个全部完全同频次。
        pool_size = min(
            len(candidates),
            NEGATIVE_COUNT
            * POOL_MULTIPLIER,
        )

        local_pool = [
            x[2]
            for x in candidates[
                :pool_size
            ]
        ]

        if len(local_pool) < NEGATIVE_COUNT:
            continue

        negatives = rng.sample(
            local_pool,
            NEGATIVE_COUNT,
        )

        candidate_items = (
            negatives
            + [target]
        )

        rng.shuffle(
            candidate_items
        )

        new_case = dict(case)

        new_case[
            "candidates"
        ] = candidate_items

        new_cases.append(
            new_case
        )

    write_jsonl(
        config.DATA_DIR
        / "eval_cases.jsonl",
        new_cases,
    )

    print(
        "cases =",
        len(new_cases)
    )

    # sanity check
    target_pops = []
    negative_pops = []

    for c in new_cases:
        target = str(c["target"])

        target_pops.append(
            math.log1p(
                popularity.get(
                    target,
                    0
                )
            )
        )

        for item in c["candidates"]:
            if str(item) == target:
                continue

            negative_pops.append(
                math.log1p(
                    popularity.get(
                        str(item),
                        0
                    )
                )
            )

    print(
        "mean target log-pop =",
        sum(target_pops)
        / len(target_pops)
    )

    print(
        "mean negative log-pop =",
        sum(negative_pops)
        / len(negative_pops)
    )


if __name__ == "__main__":
    main()
