import numpy as np
import pandas as pd

from src import config


OUT_DIR = config.RESULT_DIR / "benefit"

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
]


def topk_indices(values, k):
    values = np.asarray(values, dtype=float)

    return np.argsort(
        -values,
        kind="stable",
    )[:k]


def main():
    df = pd.read_csv(
        OUT_DIR / "predictions.csv"
    )

    rows = []

    for case_id, g in df.groupby(
        "case_id",
        sort=False,
    ):
        g = g.reset_index(drop=True)

        true_u = g[
            "utility_margin_if_aug"
        ].to_numpy(dtype=float)

        positive_u = np.maximum(
            true_u,
            0.0,
        )

        total_positive = float(
            positive_u.sum()
        )

        selectors = {
            "BenefitRF":
                g["prob_rf"].to_numpy(),

            "BenefitLR":
                g["prob_lr"].to_numpy(),

            "Tail":
                -g[
                    "item_popularity_log"
                ].to_numpy(),

            "LowKnowledge":
                -g[
                    "knowledge_score"
                ]
                .fillna(1.0)
                .to_numpy(),

            "Oracle":
                true_u,
        }

        for budget in BUDGETS:
            k = max(
                1,
                int(
                    round(
                        budget * len(g)
                    )
                ),
            )

            oracle_idx = topk_indices(
                true_u,
                k,
            )

            oracle_utility = float(
                true_u[
                    oracle_idx
                ].sum()
            )

            for name, score in (
                selectors.items()
            ):
                idx = topk_indices(
                    score,
                    k,
                )

                selected_positive = float(
                    positive_u[
                        idx
                    ].sum()
                )

                selected_true_u = float(
                    true_u[
                        idx
                    ].sum()
                )

                capture = (
                    selected_positive
                    / total_positive
                    if total_positive > 0
                    else np.nan
                )

                regret = (
                    oracle_utility
                    - selected_true_u
                )

                rows.append({
                    "case_id": case_id,
                    "router": name,
                    "budget": budget,
                    "utility_capture": capture,
                    "selected_true_utility":
                        selected_true_u,
                    "oracle_utility":
                        oracle_utility,
                    "regret": regret,
                })

    case_df = pd.DataFrame(rows)

    case_df.to_csv(
        OUT_DIR
        / "capture_case_metrics.csv",
        index=False,
    )

    summary = (
        case_df.groupby(
            [
                "router",
                "budget",
            ]
        )
        .agg(
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
        OUT_DIR
        / "capture_summary.csv",
        index=False,
    )

    print(
        "\n===== UTILITY CAPTURE ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
