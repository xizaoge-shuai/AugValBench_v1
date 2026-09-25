import random

import numpy as np
import pandas as pd

from src import config
from src.utility.evaluate_ranker import (
    expected_tie_metrics,
)


OUT = (
    config.RESULT_DIR
    / "rank_impact"
)

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
]

RANDOM_REPEATS = 30


def topk(values, k):
    values = np.asarray(
        values,
        dtype=float,
    )

    return np.argsort(
        -values,
        kind="stable",
    )[:k]


def evaluate_selection(
    case,
    selected,
):

    base = (
        case["base_score"]
        .to_numpy(
            dtype=float
        )
    )

    clean = (
        case["clean_score"]
        .to_numpy(
            dtype=float
        )
    )

    labels = (
        case["is_target"]
        .to_numpy(
            dtype=int
        )
    )

    score = base.copy()

    score[
        selected
    ] = clean[
        selected
    ]

    return expected_tie_metrics(
        score,
        labels,
    )


def selector_quality(
    case,
    selected,
):

    pwr_u = (
        case[
            "utility_pwr_if_aug"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ndcg_u = (
        case[
            "utility_ndcg_if_aug"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pwr_pos = np.maximum(
        pwr_u,
        0.0,
    )

    ndcg_pos = np.maximum(
        ndcg_u,
        0.0,
    )

    pwr_total = float(
        pwr_pos.sum()
    )

    ndcg_total = float(
        ndcg_pos.sum()
    )

    return {
        "selected_pwr_utility":
            float(
                pwr_u[
                    selected
                ].sum()
            ),

        "selected_ndcg_utility":
            float(
                ndcg_u[
                    selected
                ].sum()
            ),

        "pwr_capture":
            (
                float(
                    pwr_pos[
                        selected
                    ].sum()
                )
                / pwr_total
                if pwr_total > 0
                else np.nan
            ),

        "ndcg_capture":
            (
                float(
                    ndcg_pos[
                        selected
                    ].sum()
                )
                / ndcg_total
                if ndcg_total > 0
                else np.nan
            ),
    }


def main():

    impact = pd.read_csv(
        OUT
        / "rank_impact_predictions.csv"
    )

    impact["item_id"] = (
        impact["item_id"]
        .astype(str)
    )

    # ---------------------------------------
    # Benefit classifier OOF predictions
    # ---------------------------------------

    benefit = pd.read_csv(
        config.RESULT_DIR
        / "benefit"
        / "predictions.csv"
    )[
        [
            "case_id",
            "item_id",
            "prob_rf",
        ]
    ].copy()

    benefit["item_id"] = (
        benefit["item_id"]
        .astype(str)
    )

    # ---------------------------------------
    # Previous margin UtilityRF OOF prediction
    # ---------------------------------------

    margin = pd.read_csv(
        config.RESULT_DIR
        / "utility"
        / "ranker_predictions.csv"
    )[
        [
            "case_id",
            "item_id",
            "pred_utility_rf",
        ]
    ].copy()

    margin["item_id"] = (
        margin["item_id"]
        .astype(str)
    )

    df = (
        impact
        .merge(
            benefit,
            on=[
                "case_id",
                "item_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            margin,
            on=[
                "case_id",
                "item_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    required = [
        "prob_rf",
        "pred_utility_rf",
        "pred_hurdle_pwr",
        "pred_hurdle_ndcg",
        "pred_direct_pwr",
        "pred_direct_ndcg",
    ]

    if df[
        required
    ].isna().any().any():

        bad = (
            df[
                required
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            "Missing merged predictions:\n"
            + bad.to_string()
        )

    rows = []

    # =======================================
    # fixed-budget evaluation
    # =======================================

    for budget in BUDGETS:

        print(
            f"budget = {budget:.2f}"
        )

        for case_id, case in df.groupby(
            "case_id",
            sort=False,
        ):

            case = (
                case.reset_index(
                    drop=True
                )
            )

            n = len(case)

            k = max(
                1,
                int(
                    round(
                        budget * n
                    )
                ),
            )

            pwr_true = (
                case[
                    "utility_pwr_if_aug"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            ndcg_true = (
                case[
                    "utility_ndcg_if_aug"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            selectors = {
                "HurdlePWR":
                    case[
                        "pred_hurdle_pwr"
                    ].to_numpy(),

                "HurdleNDCG":
                    case[
                        "pred_hurdle_ndcg"
                    ].to_numpy(),

                "DirectPWR":
                    case[
                        "pred_direct_pwr"
                    ].to_numpy(),

                "DirectNDCG":
                    case[
                        "pred_direct_ndcg"
                    ].to_numpy(),

                "BenefitRF":
                    case[
                        "prob_rf"
                    ].to_numpy(),

                "UtilityMarginRF":
                    case[
                        "pred_utility_rf"
                    ].to_numpy(),

                "Tail":
                    -case[
                        "item_popularity_log"
                    ].to_numpy(),

                "LowKnowledge":
                    -case[
                        "knowledge_score"
                    ]
                    .fillna(1.0)
                    .to_numpy(),

                "OraclePWR":
                    pwr_true,

                "OracleNDCG":
                    ndcg_true,
            }

            for router, score in (
                selectors.items()
            ):

                selected = topk(
                    score,
                    k,
                )

                ranking = (
                    evaluate_selection(
                        case,
                        selected,
                    )
                )

                quality = (
                    selector_quality(
                        case,
                        selected,
                    )
                )

                rows.append({
                    "case_id":
                        case_id,
                    "budget":
                        budget,
                    "router":
                        router,
                    **ranking,
                    **quality,
                })

            # ===================================
            # Random baseline
            # ===================================

            random_rows = []

            for rep in range(
                RANDOM_REPEATS
            ):

                rng = random.Random(
                    2027
                    + int(case_id) * 1009
                    + int(
                        budget * 1000
                    ) * 17
                    + rep * 97
                )

                ids = list(
                    range(n)
                )

                rng.shuffle(ids)

                selected = np.asarray(
                    ids[:k],
                    dtype=int,
                )

                ranking = (
                    evaluate_selection(
                        case,
                        selected,
                    )
                )

                quality = (
                    selector_quality(
                        case,
                        selected,
                    )
                )

                random_rows.append({
                    **ranking,
                    **quality,
                })

            avg = {}

            for key in random_rows[0]:
                vals = [
                    x[key]
                    for x in random_rows
                ]

                avg[key] = float(
                    np.nanmean(vals)
                )

            rows.append({
                "case_id":
                    case_id,
                "budget":
                    budget,
                "router":
                    "Random",
                **avg,
            })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUT
        / "rank_impact_case_metrics.csv",
        index=False,
    )

    summary = (
        result.groupby(
            [
                "router",
                "budget",
            ]
        )
        .agg(
            ndcg10=(
                "ndcg10",
                "mean",
            ),
            hr10=(
                "hr10",
                "mean",
            ),
            mrr=(
                "mrr",
                "mean",
            ),
            rank=(
                "rank",
                "mean",
            ),
            pairwise_win_rate=(
                "pairwise_win_rate",
                "mean",
            ),
            pairwise_margin=(
                "pairwise_margin",
                "mean",
            ),
            pwr_capture=(
                "pwr_capture",
                "mean",
            ),
            ndcg_capture=(
                "ndcg_capture",
                "mean",
            ),
            selected_pwr_utility=(
                "selected_pwr_utility",
                "mean",
            ),
            selected_ndcg_utility=(
                "selected_ndcg_utility",
                "mean",
            ),
            n=(
                "case_id",
                "count",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT
        / "rank_impact_summary.csv",
        index=False,
    )

    print(
        "\n===== SAME-BUDGET RESULT ====="
    )

    for budget in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ]:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"BUDGET = {budget:.2f}"
        )

        print(
            "=" * 100
        )

        x = summary[
            summary[
                "budget"
            ] == budget
        ]

        print(
            x[
                [
                    "router",
                    "ndcg10",
                    "hr10",
                    "pairwise_win_rate",
                    "pwr_capture",
                    "ndcg_capture",
                    "selected_pwr_utility",
                    "selected_ndcg_utility",
                ]
            ]
            .sort_values(
                "ndcg10",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
