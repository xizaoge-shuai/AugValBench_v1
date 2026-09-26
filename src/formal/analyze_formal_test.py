import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config


BASE = (
    config.RESULT_DIR
    / f"formal_{config.AMAZON_CATEGORY.lower()}"
)

SELECTED_PATH = (
    BASE
    / "test_selected_case_metrics.csv"
)

CURVE_PATH = (
    BASE
    / "test_budget_case_metrics.csv"
)

SELECTION_PATH = (
    BASE
    / "selection.json"
)

BOOTSTRAP = 20000
SEED = 2027

PRIMARY_METRICS = [
    "ndcg10",
    "pairwise_win_rate",
]

SELECTED_BASELINES = [
    "NoAugment",
    "UniformClean",
    "Tail",
    "Random",
    "LowKnowledge",
    "Uncertainty",
    "BenefitRF_NDCG",
    "BenefitRF_Margin",
]

FIXED_BASELINES = [
    "BenefitRF_NDCG",
    "BenefitRF_Margin",
    "Tail",
    "Random",
    "LowKnowledge",
    "Uncertainty",
]

FIXED_BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]


def bootstrap_mean_ci(
    values,
    seed,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(x),
        size=(
            BOOTSTRAP,
            len(x),
        ),
    )

    means = x[idx].mean(
        axis=1
    )

    lo, hi = np.percentile(
        means,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(lo),
        float(hi),
    )


def paired_stats(
    method,
    baseline,
    seed,
):
    a = np.asarray(
        method,
        dtype=float,
    )

    b = np.asarray(
        baseline,
        dtype=float,
    )

    diff = a - b

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(diff),
        size=(
            BOOTSTRAP,
            len(diff),
        ),
    )

    boot_mean = diff[
        idx
    ].mean(
        axis=1
    )

    ci_low, ci_high = (
        np.percentile(
            boot_mean,
            [
                2.5,
                97.5,
            ],
        )
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

    baseline_mean = float(
        b.mean()
    )

    mean_diff = float(
        diff.mean()
    )

    relative = (
        mean_diff
        / baseline_mean
        if abs(
            baseline_mean
        ) > 1e-15
        else np.nan
    )

    return {
        "method_mean":
            float(
                a.mean()
            ),

        "baseline_mean":
            baseline_mean,

        "mean_difference":
            mean_diff,

        "relative_difference":
            relative,

        "bootstrap_ci_low":
            float(
                ci_low
            ),

        "bootstrap_ci_high":
            float(
                ci_high
            ),

        "wilcoxon_p":
            float(p),

        "positive_rate":
            float(
                np.mean(
                    diff > 0
                )
            ),

        "negative_rate":
            float(
                np.mean(
                    diff < 0
                )
            ),

        "zero_rate":
            float(
                np.mean(
                    diff == 0
                )
            ),

        "n":
            len(diff),
    }


def holm_adjust(
    pvalues,
):
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    result = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    valid = np.where(
        ~np.isnan(p)
    )[0]

    if len(valid) == 0:
        return result

    order = valid[
        np.argsort(
            p[valid]
        )
    ]

    m = len(order)

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        adjusted = (
            p[idx]
            * (
                m - rank
            )
        )

        running = max(
            running,
            adjusted,
        )

        result[idx] = min(
            running,
            1.0,
        )

    return result


def absolute_ci_table(
    df,
):
    rows = []

    for (
        router,
        budget
    ), x in df.groupby(
        [
            "router",
            "budget",
        ]
    ):

        for metric in [
            "ndcg10",
            "hr10",
            "mrr",
            "pairwise_win_rate",
        ]:

            values = x[
                metric
            ].to_numpy()

            lo, hi = (
                bootstrap_mean_ci(
                    values,
                    SEED
                    + len(rows),
                )
            )

            rows.append({
                "router":
                    router,
                "budget":
                    budget,
                "metric":
                    metric,
                "mean":
                    float(
                        values.mean()
                    ),
                "ci_low":
                    lo,
                "ci_high":
                    hi,
                "n":
                    len(values),
            })

    return pd.DataFrame(
        rows
    )


def selected_comparison(
    df,
):
    rows = []

    method_name = (
        "DirectNDCG"
    )

    method = df[
        df[
            "router"
        ] == method_name
    ].set_index(
        "case_id"
    )

    if method.empty:
        raise RuntimeError(
            "DirectNDCG missing."
        )

    for baseline_name in (
        SELECTED_BASELINES
    ):

        baseline = df[
            df[
                "router"
            ] == baseline_name
        ].set_index(
            "case_id"
        )

        if baseline.empty:
            continue

        common = (
            method.index
            .intersection(
                baseline.index
            )
        )

        for metric in (
            PRIMARY_METRICS
        ):

            stats = paired_stats(
                method.loc[
                    common,
                    metric,
                ],
                baseline.loc[
                    common,
                    metric,
                ],
                SEED
                + len(rows),
            )

            rows.append({
                "comparison":
                    "validation_selected",
                "budget":
                    float(
                        method[
                            "budget"
                        ].iloc[0]
                    ),
                "method":
                    method_name,
                "baseline":
                    baseline_name,
                "baseline_budget":
                    float(
                        baseline[
                            "budget"
                        ].iloc[0]
                    ),
                "metric":
                    metric,
                **stats,
            })

    out = pd.DataFrame(
        rows
    )

    out[
        "holm_p"
    ] = holm_adjust(
        out[
            "wilcoxon_p"
        ].to_numpy()
    )

    return out


def fixed_budget_comparison(
    df,
):
    rows = []

    for budget in FIXED_BUDGETS:

        x = df[
            np.isclose(
                df[
                    "budget"
                ],
                budget,
            )
        ]

        method = x[
            x[
                "router"
            ] == "DirectNDCG"
        ].set_index(
            "case_id"
        )

        if method.empty:
            continue

        for baseline_name in (
            FIXED_BASELINES
        ):

            baseline = x[
                x[
                    "router"
                ] == baseline_name
            ].set_index(
                "case_id"
            )

            if baseline.empty:
                continue

            common = (
                method.index
                .intersection(
                    baseline.index
                )
            )

            for metric in (
                PRIMARY_METRICS
            ):

                stats = paired_stats(
                    method.loc[
                        common,
                        metric,
                    ],
                    baseline.loc[
                        common,
                        metric,
                    ],
                    SEED
                    + 1000
                    + len(rows),
                )

                rows.append({
                    "comparison":
                        "fixed_budget",
                    "budget":
                        budget,
                    "method":
                        "DirectNDCG",
                    "baseline":
                        baseline_name,
                    "baseline_budget":
                        budget,
                    "metric":
                        metric,
                    **stats,
                })

    out = pd.DataFrame(
        rows
    )

    out[
        "holm_p"
    ] = holm_adjust(
        out[
            "wilcoxon_p"
        ].to_numpy()
    )

    return out


def main():

    selected = pd.read_csv(
        SELECTED_PATH
    )

    curve = pd.read_csv(
        CURVE_PATH
    )

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        selection = json.load(f)

    print(
        "===== SPLIT ====="
    )

    print(
        "train =",
        selection[
            "train_cases"
        ]
    )

    print(
        "val   =",
        selection[
            "val_cases"
        ]
    )

    print(
        "test  =",
        selection[
            "test_cases"
        ]
    )

    print(
        "\n===== SELECTED BUDGETS ====="
    )

    for k, v in selection[
        "selected_budgets"
    ].items():
        print(
            f"{k:20s} {v}"
        )

    absolute = (
        absolute_ci_table(
            selected
        )
    )

    selected_sig = (
        selected_comparison(
            selected
        )
    )

    fixed_sig = (
        fixed_budget_comparison(
            curve
        )
    )

    absolute.to_csv(
        BASE
        / "test_absolute_ci.csv",
        index=False,
    )

    selected_sig.to_csv(
        BASE
        / "test_selected_significance.csv",
        index=False,
    )

    fixed_sig.to_csv(
        BASE
        / "test_fixed_budget_significance.csv",
        index=False,
    )

    print(
        "\n===== VALIDATION-SELECTED POLICY: NDCG ====="
    )

    print(
        selected_sig[
            selected_sig[
                "metric"
            ] == "ndcg10"
        ][
            [
                "baseline",
                "baseline_budget",
                "method_mean",
                "baseline_mean",
                "mean_difference",
                "relative_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "wilcoxon_p",
                "holm_p",
            ]
        ]
        .sort_values(
            "mean_difference",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n===== FIXED 10% BUDGET: NDCG ====="
    )

    print(
        fixed_sig[
            (
                np.isclose(
                    fixed_sig[
                        "budget"
                    ],
                    0.10,
                )
            )
            &
            (
                fixed_sig[
                    "metric"
                ]
                == "ndcg10"
            )
        ][
            [
                "baseline",
                "method_mean",
                "baseline_mean",
                "mean_difference",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "wilcoxon_p",
                "holm_p",
            ]
        ]
        .sort_values(
            "mean_difference",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n===== DIRECT NDCG ABSOLUTE 95% CI ====="
    )

    print(
        absolute[
            (
                absolute[
                    "router"
                ] == "DirectNDCG"
            )
            &
            (
                absolute[
                    "metric"
                ] == "ndcg10"
            )
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
