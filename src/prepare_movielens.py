from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import urllib.request
import zipfile

import pandas as pd

from src import config
from src.utils import (
    set_seed,
    write_jsonl,
)


MOVIELENS_URL = (
    "https://files.grouplens.org/"
    "datasets/movielens/ml-1m.zip"
)


def iterative_kcore(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    for _ in range(
        config.MAX_KCORE_ROUNDS
    ):
        before = len(out)

        user_cnt = (
            out["user_id"]
            .value_counts()
        )

        item_cnt = (
            out["item_id"]
            .value_counts()
        )

        valid_users = user_cnt[
            user_cnt
            >= config.MIN_USER_INTERACTIONS
        ].index

        valid_items = item_cnt[
            item_cnt
            >= config.MIN_ITEM_INTERACTIONS
        ].index

        out = out[
            out["user_id"]
            .isin(valid_users)
            &
            out["item_id"]
            .isin(valid_items)
        ].copy()

        if len(out) == before:
            break

    return out


def ensure_raw_data():

    raw_dir = (
        config.DATA_DIR
        / "raw_movielens"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = (
        raw_dir
        / "ml-1m.zip"
    )

    extracted = (
        raw_dir
        / "ml-1m"
    )

    ratings_path = (
        extracted
        / "ratings.dat"
    )

    movies_path = (
        extracted
        / "movies.dat"
    )

    if (
        ratings_path.exists()
        and movies_path.exists()
    ):
        return (
            ratings_path,
            movies_path,
        )

    if not zip_path.exists():

        print(
            "Downloading MovieLens-1M..."
        )

        urllib.request.urlretrieve(
            MOVIELENS_URL,
            zip_path,
        )

    print(
        "Extracting MovieLens-1M..."
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:

        zf.extractall(
            raw_dir
        )

    if (
        not ratings_path.exists()
        or not movies_path.exists()
    ):
        raise RuntimeError(
            "MovieLens extraction failed."
        )

    return (
        ratings_path,
        movies_path,
    )


def load_ratings(path):

    rows = []

    with path.open(
        "r",
        encoding="latin-1",
    ) as f:

        for line in f:

            parts = (
                line.rstrip("\n")
                .split("::")
            )

            if len(parts) != 4:
                continue

            uid, iid, rating, ts = (
                parts
            )

            rows.append({
                "user_id":
                    str(uid),

                "item_id":
                    str(iid),

                "rating":
                    float(rating),

                "timestamp":
                    int(ts),
            })

    return pd.DataFrame(
        rows
    )


def load_movies(path):

    meta = {}

    with path.open(
        "r",
        encoding="latin-1",
    ) as f:

        for line in f:

            parts = (
                line.rstrip("\n")
                .split("::")
            )

            if len(parts) != 3:
                continue

            iid, title, genres = (
                parts
            )

            genre_text = (
                genres.replace(
                    "|",
                    ", "
                )
            )

            meta[str(iid)] = {
                "item_id":
                    str(iid),

                "title":
                    title,

                "main_category":
                    "Movie",

                "categories":
                    genre_text,

                "features":
                    genre_text,

                "description":
                    "",

                "store":
                    "MovieLens",

                "details":
                    (
                        f"Genres: "
                        f"{genre_text}"
                    ),
            }

    return meta


def main():

    set_seed(
        config.RANDOM_SEED
    )

    rng = random.Random(
        config.RANDOM_SEED
    )

    config.DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== MovieLens-1M ====="
    )

    ratings_path, movies_path = (
        ensure_raw_data()
    )

    print(
        "[1/5] Loading ratings..."
    )

    df = load_ratings(
        ratings_path
    )

    print(
        "raw ratings:",
        len(df)
    )

    # 与 Amazon 实验一致：
    # 将 >=4 视为 positive feedback
    df = df[
        df["rating"]
        >= config.POSITIVE_RATING
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
        ],
        keep="last",
    )

    print(
        "positive ratings:",
        len(df)
    )

    print(
        "[2/5] Iterative k-core..."
    )

    df = iterative_kcore(
        df
    )

    print(
        "after k-core:",
        len(df)
    )

    if df.empty:
        raise RuntimeError(
            "MovieLens k-core is empty."
        )

    df = df.sort_values(
        [
            "user_id",
            "timestamp",
        ]
    )

    user_all_items = (
        defaultdict(set)
    )

    train_rows = []
    raw_cases = []

    for uid, g in df.groupby(
        "user_id",
        sort=False,
    ):

        seq = (
            g["item_id"]
            .astype(str)
            .tolist()
        )

        if len(seq) < 3:
            continue

        target = seq[-1]
        history = seq[:-1]

        user_all_items[
            str(uid)
        ] = set(seq)

        train_rows.append({
            "user_id":
                str(uid),

            "items":
                history,
        })

        raw_cases.append({
            "user_id":
                str(uid),

            "history":
                history,

            "target":
                target,
        })

    pop = Counter()

    for row in train_rows:
        pop.update(
            row["items"]
        )

    item_universe = sorted(
        pop.keys()
    )

    rng.shuffle(
        raw_cases
    )

    max_users = (
        config.DEBUG_USERS
        if config.DEBUG
        else config.MAX_EVAL_USERS
    )

    raw_cases = raw_cases[
        :max_users
    ]

    print(
        "[3/5] Building "
        "uniform candidate sets..."
    )

    cases = []

    for idx, r in enumerate(
        raw_cases
    ):

        forbidden = user_all_items[
            r["user_id"]
        ]

        neg_pool = [
            item
            for item
            in item_universe
            if (
                item not in forbidden
                and item
                != r["target"]
            )
        ]

        if (
            len(neg_pool)
            < config.NEGATIVE_COUNT
        ):
            continue

        negatives = rng.sample(
            neg_pool,
            config.NEGATIVE_COUNT,
        )

        candidates = (
            negatives
            + [
                r["target"]
            ]
        )

        rng.shuffle(
            candidates
        )

        cases.append({
            "case_id":
                idx,

            "user_id":
                r["user_id"],

            "history":
                r["history"][
                    -config.MAX_HISTORY:
                ],

            "target":
                r["target"],

            "candidates":
                candidates,
        })

    print(
        "[4/5] Loading movie metadata..."
    )

    all_meta = load_movies(
        movies_path
    )

    needed = set(
        item_universe
    )

    for c in cases:
        needed.add(
            c["target"]
        )

        needed.update(
            c["history"]
        )

        needed.update(
            c["candidates"]
        )

    meta = {
        iid: all_meta[iid]
        for iid in needed
        if iid in all_meta
    }

    kept = []

    for c in cases:

        if (
            c["target"]
            not in meta
        ):
            continue

        history = [
            x
            for x in c["history"]
            if x in meta
        ]

        candidates = [
            x
            for x in c["candidates"]
            if x in meta
        ]

        if (
            len(history) < 2
            or c["target"]
            not in candidates
            or len(candidates) < 5
        ):
            continue

        c["history"] = history
        c["candidates"] = (
            candidates
        )

        kept.append(c)

    # Reassign case_id so it is dense.
    for idx, c in enumerate(
        kept
    ):
        c["case_id"] = idx

    print(
        "[5/5] Saving..."
    )

    write_jsonl(
        config.DATA_DIR
        / "train_sequences.jsonl",
        train_rows,
    )

    write_jsonl(
        config.DATA_DIR
        / "eval_cases.jsonl",
        kept,
    )

    with (
        config.DATA_DIR
        / "item_popularity.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            pop,
            f,
            ensure_ascii=False,
        )

    with (
        config.DATA_DIR
        / "item_meta.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            meta,
            f,
            ensure_ascii=False,
        )

    print(
        "\n===== DONE ====="
    )

    print(
        "train users:",
        len(train_rows)
    )

    print(
        "eval cases:",
        len(kept)
    )

    print(
        "items:",
        len(item_universe)
    )

    print(
        "metadata:",
        len(meta)
    )

    print(
        "saved to:",
        config.DATA_DIR
    )


if __name__ == "__main__":
    main()
