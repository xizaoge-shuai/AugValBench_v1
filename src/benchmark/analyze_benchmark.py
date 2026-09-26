import json

import numpy as np
import pandas as pd

from src import config
from src.utils import read_jsonl


CONDITION_ORDER = [
    "base",
    "clean",
    "redundant",
    "partial",
    "irrelevant",
    "conflict",
]

STRATUM_ORDER = [
    "P1_Tail",
    "P2_MidTail",
    "P3_MidHead",
    "P4_Head",
]


def popularity_stratum(pct):

    if pct <= 0.25:
        return "P1_Tail"

    if pct <= 0.50:
        return "P2_MidTail"

    if pct <= 0.75:
        return "P3_MidHead"

    return "P4_Head"


def summarize(df, group_cols):

    rows = []

    for keys, g in df.groupby(
        group_cols,
        sort=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (keys,)

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row.update({
            "n":
                g[
                    "case_id"
                ].nunique(),

            "ndcg10":
                g[
                    "ndcg10"
                ].mean(),

            "delta_ndcg10":
                g[
                    "delta_ndcg10"
                ].mean(),

            "hr10":
                g[
                    "hr10"
                ].mean(),

            "mrr":
                g[
                    "mrr"
                ].mean(),

            "pairwise_win_rate":
                g[
                    "pairwise_win_rate"
                ].mean(),

            "delta_pairwise_win_rate":
                g[
                    "delta_pairwise_win_rate"
                ].mean(),

            "pairwise_margin":
                g[
                    "pairwise_margin"
                ].mean(),

            "delta_pairwise_margin":
                g[
                    "delta_pairwise_margin"
                ].mean(),

            "ndcg_harmful_rate":
                (
                    g[
                        "delta_ndcg10"
                    ] < 0
                ).mean(),

            "ndcg_helpful_rate":
                (
                    g[
                        "delta_ndcg10"
                    ] > 0
                ).mean(),

            "pwr_harmful_rate":
                (
                    g[
                        "delta_pairwise_win_rate"
                    ] < 0
                ).mean(),

            "pwr_helpful_rate":
                (
                    g[
                        "delta_pairwise_win_rate"
                    ] > 0
                ).mean(),
        })

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def main():

    result_dir = (
        config.RESULT_DIR
    )

    metrics = pd.read_csv(
        result_dir
        / "case_metrics.csv"
    )

    cases = read_jsonl(
        config.DATA_DIR
        / "eval_cases.jsonl"
    )

    with (
        config.DATA_DIR
        / "item_popularity.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:

        popularity = json.load(f)

    case_info = pd.DataFrame([
        {
            "case_id":
                int(c["case_id"]),

            "target":
                str(c["target"]),

            "target_popularity":
                int(
                    popularity.get(
                        str(c["target"]),
                        0,
                    )
                ),
        }
        for c in cases
    ])

    # Preserve ties:
    # identical popularity gets identical percentile.
    case_info[
        "popularity_percentile"
    ] = (
        case_info[
            "target_popularity"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    case_info[
        "popularity_stratum"
    ] = (
        case_info[
            "popularity_percentile"
        ]
        .apply(
            popularity_stratum
        )
    )

    metrics = metrics.merge(
        case_info,
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    base = (
        metrics[
            metrics[
                "condition"
            ] == "base"
        ][
            [
                "case_id",
                "ndcg10",
                "pairwise_win_rate",
                "pairwise_margin",
            ]
        ]
        .rename(
            columns={
                "ndcg10":
                    "base_ndcg10",

                "pairwise_win_rate":
                    "base_pairwise_win_rate",

                "pairwise_margin":
                    "base_pairwise_margin",
            }
        )
    )

    metrics = metrics.merge(
        base,
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    metrics[
        "delta_ndcg10"
    ] = (
        metrics["ndcg10"]
        - metrics["base_ndcg10"]
    )

    metrics[
        "delta_pairwise_win_rate"
    ] = (
        metrics[
            "pairwise_win_rate"
        ]
        - metrics[
            "base_pairwise_win_rate"
        ]
    )

    metrics[
        "delta_pairwise_margin"
    ] = (
        metrics[
            "pairwise_margin"
        ]
        - metrics[
            "base_pairwise_margin"
        ]
    )

    global_summary = summarize(
        metrics,
        [
            "condition",
        ],
    )

    by_popularity = summarize(
        metrics,
        [
            "condition",
            "popularity_stratum",
        ],
    )

    condition_rank = {
        x: i
        for i, x in enumerate(
            CONDITION_ORDER
        )
    }

    stratum_rank = {
        x: i
        for i, x in enumerate(
            STRATUM_ORDER
        )
    }

    global_summary[
        "_condition_order"
    ] = (
        global_summary[
            "condition"
        ]
        .map(
            condition_rank
        )
    )

    global_summary = (
        global_summary
        .sort_values(
            "_condition_order"
        )
        .drop(
            columns=[
                "_condition_order"
            ]
        )
    )

    by_popularity[
        "_condition_order"
    ] = (
        by_popularity[
            "condition"
        ]
        .map(
            condition_rank
        )
    )

    by_popularity[
        "_stratum_order"
    ] = (
        by_popularity[
            "popularity_stratum"
        ]
        .map(
            stratum_rank
        )
    )

    by_popularity = (
        by_popularity
        .sort_values(
            [
                "_condition_order",
                "_stratum_order",
            ]
        )
        .drop(
            columns=[
                "_condition_order",
                "_stratum_order",
            ]
        )
    )

    metrics.to_csv(
        result_dir
        / "benchmark_case_table.csv",
        index=False,
    )

    global_summary.to_csv(
        result_dir
        / "benchmark_summary.csv",
        index=False,
    )

    by_popularity.to_csv(
        result_dir
        / "benchmark_by_popularity.csv",
        index=False,
    )

    pop_stats = (
        case_info.groupby(
            "popularity_stratum"
        )
        .agg(
            n=(
                "case_id",
                "count",
            ),
            pop_min=(
                "target_popularity",
                "min",
            ),
            pop_median=(
                "target_popularity",
                "median",
            ),
            pop_mean=(
                "target_popularity",
                "mean",
            ),
            pop_max=(
                "target_popularity",
                "max",
            ),
        )
        .reset_index()
    )

    pop_stats.to_csv(
        result_dir
        / "popularity_strata.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

        "conditions":
            CONDITION_ORDER,

        "num_cases":
            int(
                case_info[
                    "case_id"
                ].nunique()
            ),
    }

    with (
        result_dir
        / "benchmark_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n===== AUGVALBENCH GLOBAL ====="
    )

    print(
        global_summary.to_string(
            index=False
        )
    )

    print(
        "\n===== POPULARITY STRATA ====="
    )

    print(
        pop_stats.to_string(
            index=False
        )
    )

    print(
        "\n===== CONDITION x POPULARITY ====="
    )

    print(
        by_popularity.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
