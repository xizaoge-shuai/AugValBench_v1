import csv
import json
import random
from pathlib import Path

from src import config
from src.utils import read_jsonl, write_jsonl


SEED = 2027

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20


def main():
    input_path = (
        config.DATA_DIR
        / "eval_cases.jsonl"
    )

    cases = read_jsonl(
        input_path
    )

    if not cases:
        raise RuntimeError(
            "No evaluation cases found."
        )

    users = sorted(
        {
            str(c["user_id"])
            for c in cases
        }
    )

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        users
    )

    n_users = len(users)

    n_train = int(
        n_users * TRAIN_RATIO
    )

    n_val = int(
        n_users * VAL_RATIO
    )

    train_users = set(
        users[:n_train]
    )

    val_users = set(
        users[
            n_train:
            n_train + n_val
        ]
    )

    test_users = set(
        users[
            n_train + n_val:
        ]
    )

    if (
        train_users & val_users
        or train_users & test_users
        or val_users & test_users
    ):
        raise RuntimeError(
            "User leakage between splits."
        )

    split_dir = (
        config.DATA_DIR
        / "formal"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_cases = {
        "train": [],
        "val": [],
        "test": [],
    }

    manifest = []

    for c in cases:
        uid = str(
            c["user_id"]
        )

        if uid in train_users:
            split = "train"

        elif uid in val_users:
            split = "val"

        elif uid in test_users:
            split = "test"

        else:
            raise RuntimeError(
                f"Unknown user: {uid}"
            )

        split_cases[
            split
        ].append(c)

        manifest.append({
            "case_id":
                int(c["case_id"]),
            "user_id":
                uid,
            "split":
                split,
        })

    for split, rows in (
        split_cases.items()
    ):
        write_jsonl(
            split_dir
            / f"eval_{split}.jsonl",
            rows,
        )

    manifest_path = (
        split_dir
        / "split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "user_id",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest
        )

    print(
        "\n===== FORMAL SPLIT ====="
    )

    print(
        "users total =",
        n_users
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        rows = split_cases[
            split
        ]

        print(
            f"{split:5s}: "
            f"cases={len(rows)}"
        )

    print(
        "\nmanifest =",
        manifest_path
    )


if __name__ == "__main__":
    main()
