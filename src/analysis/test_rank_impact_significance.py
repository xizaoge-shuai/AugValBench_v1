import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from src import config


INPUT = (
    config.RESULT_DIR
    / "rank_impact"
    / "rank_impact_case_metrics.csv"
)

OUTPUT = (
    config.RESULT_DIR
    / "rank_impact"
    / "significance.csv"
)

BUDGETS = [
    0.10,
    0.15,
    0.20,
]

METHOD = "DirectNDCG"

BASELINES = [
    "Tail",
    "BenefitRF",
    "HurdlePWR",
    "UtilityMarginRF",
    "Random",
    "LowKnowledge",
]

METRICS = [
    "ndcg10",
    "pairwise_win_rate",
]

BOOTSTRAP = 10000
SEED = 2027


def bootstrap_ci(diff, n_boot=BOOTSTRAP):
    diff = np.asarray(
        diff,
        dtype=float,
    )

    rng = np.random.default_rng(
        SEED
    )

    n = len(diff)

    samples = rng.choice(
        diff,
        size=(n_boot, n),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    lo, hi = np.percentile(
        means,
        [2.5, 97.5],
    )

    return (
        float(lo),
        float(hi),
    )


def paired_test(a, b):
    diff = (
        np.asarray(a, dtype=float)
        - np.asarray(b, dtype=float)
    )

    mean_diff = float(
        diff.mean()
    )

    median_diff = float(
        np.median(diff)
    )

    positive_rate = float(
        np.mean(diff > 0)
    )

    negative_rate = float(
        np.mean(diff < 0)
    )

    zero_rate = float(
        np.mean(diff == 0)
    )

    ci_low, ci_high = (
        bootstrap_ci(diff)
    )

    nonzero = diff[
        diff != 0
    ]

    if len(nonzero) >= 2:
        _, p = wilcoxon(
            nonzero,
            alternative="two-sided",
        )
    else:
        p = np.nan

    return {
        "mean_difference":
            mean_diff,
        "median_difference":
            median_diff,
        "bootstrap_ci_low":
            ci_low,
        "bootstrap_ci_high":
            ci_high,
        "wilcoxon_p":
            float(p),
        "positive_rate":
            positive_rate,
        "negative_rate":
            negative_rate,
        "zero_rate":
            zero_rate,
        "n":
            len(diff),
    }


def main():
    df = pd.read_csv(
        INPUT
    )

    print(
        "rows =",
        len(df)
    )

    print(
        "cases =",
        df["case_id"].nunique()
    )

    rows = []

    for budget in BUDGETS:

        x = df[
            df["budget"] == budget
        ]

        for metric in METRICS:

            pivot = x.pivot(
                index="case_id",
                columns="router",
                values=metric,
            )

            if METHOD not in pivot:
                raise RuntimeError(
                    f"{METHOD} missing"
                )

            for baseline in BASELINES:

                if baseline not in pivot:
                    continue

                pair = (
                    pivot[
                        [
                            METHOD,
                            baseline,
                        ]
                    ]
                    .dropna()
                )

                stats = paired_test(
                    pair[METHOD],
                    pair[baseline],
                )

                rows.append({
                    "budget":
                        budget,
                    "method":
                        METHOD,
                    "baseline":
                        baseline,
                    "metric":
                        metric,
                    **stats,
                })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "\n===== SIGNIFICANCE ====="
    )

    print(
        out.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()
