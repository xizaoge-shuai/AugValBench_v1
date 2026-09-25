import random

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config
from src.utils import (
    dcg_single_positive,
    hit_single_positive,
    mrr_single_positive,
)


OUT_DIR = (
    config.RESULT_DIR
    / "benefit"
)

BUDGET_RATIO = 0.50


def metrics(df):
    x = df.sort_values(
        "chosen_score",
        ascending=False,
        kind="stable",
    ).reset_index(
        drop=True
    )

    target = x[
        x["is_target"] == 1
    ]

    if len(target) != 1:
        return None

    rank = (
        target.index[0]
        + 1
    )

    target_score = float(
        target.iloc[0][
            "chosen_score"
        ]
    )

    neg = (
        x.loc[
            x["is_target"] == 0,
            "chosen_score",
        ]
        .to_numpy(
            dtype=float
        )
    )

    wins = (
        target_score > neg
    ).astype(float)

    ties = (
        target_score == neg
    ).astype(float)

    pwr = float(
        np.mean(
            wins
            + 0.5 * ties
        )
    )

    margin = float(
        target_score
        - np.mean(neg)
    )

    return {
        "ndcg10": dcg_single_positive(
            rank,
            10,
        ),
        "hr10": hit_single_positive(
            rank,
            10,
        ),
        "mrr": mrr_single_positive(
            rank,
        ),
        "rank": rank,
        "pairwise_win_rate": pwr,
        "pairwise_margin": margin,
    }


def select_top(
    values,
    k,
    largest=True,
):
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

    mask = np.zeros(
        len(values),
        dtype=bool,
    )

    mask[
        order[:k]
    ] = True

    return mask


def main():
    df = pd.read_csv(
        OUT_DIR
        / "predictions.csv"
    )

    rng = random.Random(
        2027
    )

    rows = []

    for case_id, g0 in df.groupby(
        "case_id",
        sort=False,
    ):
        g0 = g0.reset_index(
            drop=True
        ).copy()

        n = len(g0)

        k = max(
            1,
            int(
                round(
                    n
                    * BUDGET_RATIO
                )
            ),
        )

        variants = {}

        variants[
            "NoAugment"
        ] = np.zeros(
            n,
            dtype=bool,
        )

        variants[
            "UniformClean"
        ] = np.ones(
            n,
            dtype=bool,
        )

        # fair 50% random budget
        random_idx = list(
            range(n)
        )

        rng.shuffle(
            random_idx
        )

        mask = np.zeros(
            n,
            dtype=bool,
        )

        mask[
            random_idx[:k]
        ] = True

        variants[
            "Random50"
        ] = mask

        # low popularity -> augment
        variants[
            "Tail50"
        ] = select_top(
            g0[
                "item_popularity_log"
            ].to_numpy(),
            k,
            largest=False,
        )

        # low knowledge -> augment
        knowledge = (
            g0[
                "knowledge_score"
            ]
            .fillna(1.0)
            .to_numpy()
        )

        variants[
            "LowKnowledge50"
        ] = select_top(
            knowledge,
            k,
            largest=False,
        )

        variants[
            "BenefitLR50"
        ] = select_top(
            g0[
                "prob_lr"
            ].to_numpy(),
            k,
            largest=True,
        )

        variants[
            "BenefitRF50"
        ] = select_top(
            g0[
                "prob_rf"
            ].to_numpy(),
            k,
            largest=True,
        )

        # 使用真实 counterfactual utility，
        # 仅作为 diagnostic oracle。
        variants[
            "LabelOracle50"
        ] = select_top(
            g0[
                "utility_margin_if_aug"
            ].to_numpy(),
            k,
            largest=True,
        )

        for name, mask in (
            variants.items()
        ):
            x = g0.copy()

            x[
                "chosen_score"
            ] = np.where(
                mask,
                x["clean_score"],
                x["base_score"],
            )

            m = metrics(
                x
            )

            if m is None:
                continue

            rows.append({
                "case_id": case_id,
                "router": name,
                "augment_ratio": float(
                    mask.mean()
                ),
                **m,
            })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUT_DIR
        / "router_case_metrics.csv",
        index=False,
    )

    summary = (
        out.groupby(
            "router"
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
            augment_ratio=(
                "augment_ratio",
                "mean",
            ),
            n=(
                "case_id",
                "count",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "ndcg10",
        ascending=False,
    )

    summary.to_csv(
        OUT_DIR
        / "router_summary.csv",
        index=False,
    )

    print(
        "\n===== ROUTER SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # paired significance
    pivot_ndcg = out.pivot(
        index="case_id",
        columns="router",
        values="ndcg10",
    )

    pivot_pwr = out.pivot(
        index="case_id",
        columns="router",
        values="pairwise_win_rate",
    )

    tests = []

    for router in [
        "BenefitLR50",
        "BenefitRF50",
    ]:
        for baseline in [
            "NoAugment",
            "UniformClean",
            "Random50",
            "Tail50",
            "LowKnowledge50",
        ]:
            for metric_name, pivot in [
                (
                    "ndcg10",
                    pivot_ndcg,
                ),
                (
                    "pairwise_win_rate",
                    pivot_pwr,
                ),
            ]:
                diff = (
                    pivot[router]
                    - pivot[baseline]
                ).dropna()

                nonzero = diff[
                    diff != 0
                ]

                if len(nonzero) >= 2:
                    stat, p = wilcoxon(
                        nonzero
                    )
                else:
                    stat = np.nan
                    p = np.nan

                tests.append({
                    "router": router,
                    "baseline": baseline,
                    "metric": metric_name,
                    "mean_difference": (
                        diff.mean()
                    ),
                    "wilcoxon_p": p,
                    "n": len(diff),
                })

    test_df = pd.DataFrame(
        tests
    )

    test_df.to_csv(
        OUT_DIR
        / "router_significance.csv",
        index=False,
    )

    print(
        "\n===== BENEFIT ROUTER "
        "SIGNIFICANCE ====="
    )

    print(
        test_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
