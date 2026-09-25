import random

import numpy as np
import pandas as pd

from src import config


OUT = config.RESULT_DIR / "utility"

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

RANDOM_REPEATS = 30


def topk(values, k, largest=True):

    values = np.asarray(
        values,
        dtype=float,
    )

    if largest:
        order = np.argsort(
            -values,
            kind="stable",
        )
    else:
        order = np.argsort(
            values,
            kind="stable",
        )

    return order[:k]


def expected_tie_metrics(
    scores,
    labels,
):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    target_indices = np.where(
        labels == 1
    )[0]

    if len(target_indices) != 1:
        raise ValueError(
            "Each case must contain exactly one target."
        )

    target_idx = int(
        target_indices[0]
    )

    target_score = scores[
        target_idx
    ]

    higher = int(
        np.sum(
            scores > target_score
        )
    )

    tied_total = int(
        np.sum(
            scores == target_score
        )
    )

    first_rank = higher + 1
    last_rank = higher + tied_total

    positions = np.arange(
        first_rank,
        last_rank + 1,
        dtype=float,
    )

    expected_rank = float(
        positions.mean()
    )

    ndcg_values = np.where(
        positions <= 10,
        1.0
        / np.log2(
            positions + 1.0
        ),
        0.0,
    )

    ndcg10 = float(
        ndcg_values.mean()
    )

    hr10 = float(
        np.mean(
            positions <= 10
        )
    )

    mrr = float(
        np.mean(
            1.0 / positions
        )
    )

    neg_scores = scores[
        labels == 0
    ]

    wins = (
        target_score
        > neg_scores
    ).astype(float)

    ties = (
        target_score
        == neg_scores
    ).astype(float)

    pwr = float(
        np.mean(
            wins
            + 0.5 * ties
        )
    )

    margin = float(
        target_score
        - np.mean(
            neg_scores
        )
    )

    return {
        "ndcg10": ndcg10,
        "hr10": hr10,
        "mrr": mrr,
        "rank": expected_rank,
        "pairwise_win_rate": pwr,
        "pairwise_margin": margin,
    }


def evaluate_selection(
    case,
    selected_idx,
):

    base = (
        case["base_score"]
        .to_numpy(dtype=float)
    )

    clean = (
        case["clean_score"]
        .to_numpy(dtype=float)
    )

    labels = (
        case["is_target"]
        .to_numpy(dtype=int)
    )

    scores = base.copy()

    scores[
        selected_idx
    ] = clean[
        selected_idx
    ]

    return expected_tie_metrics(
        scores,
        labels,
    )


def main():

    utility = pd.read_csv(
        OUT
        / "ranker_predictions.csv"
    )

    benefit = pd.read_csv(
        config.RESULT_DIR
        / "benefit"
        / "predictions.csv"
    )[
        [
            "case_id",
            "item_id",
            "prob_rf",
            "prob_lr",
        ]
    ].copy()

    utility["item_id"] = (
        utility["item_id"]
        .astype(str)
    )

    benefit["item_id"] = (
        benefit["item_id"]
        .astype(str)
    )

    df = utility.merge(
        benefit,
        on=[
            "case_id",
            "item_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if df[
        [
            "prob_rf",
            "prob_lr",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Benefit prediction merge produced missing values."
        )

    rows = []

    # ------------------------------------------------
    # NoAugment
    # ------------------------------------------------

    for case_id, case in df.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        m = evaluate_selection(
            case,
            np.array(
                [],
                dtype=int,
            ),
        )

        rows.append({
            "case_id": case_id,
            "router": "NoAugment",
            "budget": 0.0,
            "utility_capture": 0.0,
            "selected_true_utility": 0.0,
            "regret": np.nan,
            **m,
        })

    # ------------------------------------------------
    # Budgeted routing
    # ------------------------------------------------

    for budget in BUDGETS:

        print(
            f"budget = {budget:.2f}"
        )

        for case_id, case in df.groupby(
            "case_id",
            sort=False,
        ):

            case = case.reset_index(
                drop=True
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

            true_utility = (
                case[
                    "utility_margin_if_aug"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            positive_utility = np.maximum(
                true_utility,
                0.0,
            )

            total_positive = float(
                positive_utility.sum()
            )

            oracle_idx = topk(
                true_utility,
                k,
                largest=True,
            )

            oracle_signed_utility = float(
                true_utility[
                    oracle_idx
                ].sum()
            )

            selectors = {
                "UtilityRF":
                    case[
                        "pred_utility_rf"
                    ].to_numpy(),

                "UtilityGBR":
                    case[
                        "pred_utility_gbr"
                    ].to_numpy(),

                "BenefitRF":
                    case[
                        "prob_rf"
                    ].to_numpy(),

                "BenefitLR":
                    case[
                        "prob_lr"
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

                "Oracle":
                    true_utility,
            }

            for router, selector_score in (
                selectors.items()
            ):

                selected_idx = topk(
                    selector_score,
                    k,
                    largest=True,
                )

                m = evaluate_selection(
                    case,
                    selected_idx,
                )

                selected_positive = float(
                    positive_utility[
                        selected_idx
                    ].sum()
                )

                selected_signed = float(
                    true_utility[
                        selected_idx
                    ].sum()
                )

                capture = (
                    selected_positive
                    / total_positive
                    if total_positive > 0
                    else np.nan
                )

                regret = (
                    oracle_signed_utility
                    - selected_signed
                )

                rows.append({
                    "case_id": case_id,
                    "router": router,
                    "budget": budget,
                    "utility_capture":
                        capture,
                    "selected_true_utility":
                        selected_signed,
                    "regret":
                        regret,
                    **m,
                })

            # ----------------------------------------
            # Random baseline
            # ----------------------------------------

            random_metrics = []

            for rep in range(
                RANDOM_REPEATS
            ):

                rng = random.Random(
                    2027
                    + int(case_id) * 1009
                    + rep * 97
                    + int(
                        budget * 1000
                    )
                )

                ids = list(
                    range(n)
                )

                rng.shuffle(ids)

                selected_idx = np.asarray(
                    ids[:k],
                    dtype=int,
                )

                m = evaluate_selection(
                    case,
                    selected_idx,
                )

                selected_positive = float(
                    positive_utility[
                        selected_idx
                    ].sum()
                )

                selected_signed = float(
                    true_utility[
                        selected_idx
                    ].sum()
                )

                capture = (
                    selected_positive
                    / total_positive
                    if total_positive > 0
                    else np.nan
                )

                regret = (
                    oracle_signed_utility
                    - selected_signed
                )

                random_metrics.append({
                    **m,
                    "utility_capture":
                        capture,
                    "selected_true_utility":
                        selected_signed,
                    "regret":
                        regret,
                })

            avg = {
                key: float(
                    np.nanmean(
                        [
                            x[key]
                            for x
                            in random_metrics
                        ]
                    )
                )
                for key
                in random_metrics[0]
            }

            rows.append({
                "case_id": case_id,
                "router": "Random",
                "budget": budget,
                **avg,
            })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUT
        / "ranker_budget_case.csv",
        index=False,
    )

    summary = (
        result.groupby(
            [
                "router",
                "budget",
            ],
            dropna=False,
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
            utility_capture=(
                "utility_capture",
                "mean",
            ),
            selected_true_utility=(
                "selected_true_utility",
                "mean",
            ),
            regret=(
                "regret",
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
        / "ranker_budget_summary.csv",
        index=False,
    )

    print(
        "\n===== SAME-BUDGET SUMMARY ====="
    )

    for budget in BUDGETS:

        x = summary[
            summary["budget"]
            == budget
        ].copy()

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"BUDGET = {budget:.2f}"
        )

        print(
            "=" * 80
        )

        print(
            x[
                [
                    "router",
                    "ndcg10",
                    "pairwise_win_rate",
                    "utility_capture",
                    "selected_true_utility",
                    "regret",
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
