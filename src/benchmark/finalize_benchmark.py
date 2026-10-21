from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, spearmanr

from src.utils import read_jsonl
from src.utility.evaluate_ranker import expected_tie_metrics


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

BENCH_ROOT = (
    ROOT
    / "benchmark_runs"
)

FINAL_DIR = (
    BENCH_ROOT
    / "final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CONDITIONS = [
    "base",
    "clean",
    "partial",
    "redundant",
    "irrelevant",
    "conflict",
]

PERTURBATIONS = [
    "clean",
    "partial",
    "redundant",
    "irrelevant",
    "conflict",
]

STRATA = [
    "P1_Tail",
    "P2_MidTail",
    "P3_MidHead",
    "P4_Head",
]

BOOTSTRAP = 10000
SEED = 2027


def short_model(name):
    return (
        str(name)
        .split("/")[-1]
    )


def popularity_stratum(pct):

    if pct <= 0.25:
        return "P1_Tail"

    if pct <= 0.50:
        return "P2_MidTail"

    if pct <= 0.75:
        return "P3_MidHead"

    return "P4_Head"


def bootstrap_mean_ci(
    diff,
    seed,
):
    x = np.asarray(
        diff,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    means = []

    remaining = BOOTSTRAP
    batch_size = 1000

    while remaining > 0:

        b = min(
            batch_size,
            remaining,
        )

        idx = rng.integers(
            0,
            len(x),
            size=(
                b,
                len(x),
            ),
        )

        means.append(
            x[idx].mean(
                axis=1
            )
        )

        remaining -= b

    means = np.concatenate(
        means
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
    diff,
    seed,
):
    x = np.asarray(
        diff,
        dtype=float,
    )

    lo, hi = bootstrap_mean_ci(
        x,
        seed,
    )

    nonzero = x[
        np.abs(x) > 1e-15
    ]

    if len(nonzero) >= 2:

        _, p = wilcoxon(
            nonzero,
            alternative="two-sided",
        )

        p = float(p)

    else:
        p = np.nan

    return {
        "mean_delta":
            float(
                x.mean()
            ),

        "median_delta":
            float(
                np.median(x)
            ),

        "ci_low":
            lo,

        "ci_high":
            hi,

        "wilcoxon_p":
            p,

        "harm_rate":
            float(
                np.mean(
                    x < -1e-15
                )
            ),

        "help_rate":
            float(
                np.mean(
                    x > 1e-15
                )
            ),

        "zero_rate":
            float(
                np.mean(
                    np.abs(x)
                    <= 1e-15
                )
            ),

        "n":
            int(
                len(x)
            ),
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
        np.isfinite(p)
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


def summarize(
    df,
    group_cols,
):

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
                int(
                    g[
                        "case_id"
                    ].nunique()
                ),

            "ndcg10":
                float(
                    g[
                        "ndcg10"
                    ].mean()
                ),

            "delta_ndcg10":
                float(
                    g[
                        "delta_ndcg10"
                    ].mean()
                ),

            "hr10":
                float(
                    g[
                        "hr10"
                    ].mean()
                ),

            "mrr":
                float(
                    g[
                        "mrr"
                    ].mean()
                ),

            "pairwise_win_rate":
                float(
                    g[
                        "pairwise_win_rate"
                    ].mean()
                ),

            "delta_pairwise_win_rate":
                float(
                    g[
                        "delta_pairwise_win_rate"
                    ].mean()
                ),

            "pairwise_margin":
                float(
                    g[
                        "pairwise_margin"
                    ].mean()
                ),

            "delta_pairwise_margin":
                float(
                    g[
                        "delta_pairwise_margin"
                    ].mean()
                ),

            "ndcg_harmful_rate":
                float(
                    (
                        g[
                            "delta_ndcg10"
                        ]
                        < -1e-15
                    ).mean()
                ),

            "ndcg_helpful_rate":
                float(
                    (
                        g[
                            "delta_ndcg10"
                        ]
                        > 1e-15
                    ).mean()
                ),

            "pwr_harmful_rate":
                float(
                    (
                        g[
                            "delta_pairwise_win_rate"
                        ]
                        < -1e-15
                    ).mean()
                ),

            "pwr_helpful_rate":
                float(
                    (
                        g[
                            "delta_pairwise_win_rate"
                        ]
                        > 1e-15
                    ).mean()
                ),
        })

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_case_info(
    data_dir,
):

    cases = read_jsonl(
        data_dir
        / "eval_cases.jsonl"
    )

    with (
        data_dir
        / "item_popularity.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:

        popularity = json.load(f)

    info = pd.DataFrame([
        {
            "case_id":
                int(c["case_id"]),

            "target_item_id":
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

    info[
        "popularity_percentile"
    ] = (
        info[
            "target_popularity"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    info[
        "popularity_stratum"
    ] = (
        info[
            "popularity_percentile"
        ]
        .apply(
            popularity_stratum
        )
    )

    return info


def recompute_case_metrics(
    result_dir,
):

    scores = pd.read_csv(
        result_dir
        / "rerank_scores.csv"
    )

    required = {
        "case_id",
        "user_id",
        "condition",
        "is_target",
        "score",
    }

    missing = (
        required
        - set(
            scores.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{result_dir}: "
            f"missing columns {missing}"
        )

    rows = []

    for (
        case_id,
        condition
    ), g in scores.groupby(
        [
            "case_id",
            "condition",
        ],
        sort=False,
    ):

        labels = (
            g[
                "is_target"
            ]
            .to_numpy(
                dtype=int
            )
        )

        model_scores = (
            g[
                "score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if labels.sum() != 1:
            raise RuntimeError(
                f"case={case_id}, "
                f"condition={condition}: "
                f"target count={labels.sum()}"
            )

        m = expected_tie_metrics(
            model_scores,
            labels,
        )

        rows.append({
            "case_id":
                int(case_id),

            "user_id":
                str(
                    g[
                        "user_id"
                    ].iloc[0]
                ),

            "condition":
                condition,

            "target_rank":
                float(
                    m["rank"]
                ),

            "ndcg10":
                float(
                    m["ndcg10"]
                ),

            "hr10":
                float(
                    m["hr10"]
                ),

            "mrr":
                float(
                    m["mrr"]
                ),

            "pairwise_win_rate":
                float(
                    m[
                        "pairwise_win_rate"
                    ]
                ),

            "pairwise_margin":
                float(
                    m[
                        "pairwise_margin"
                    ]
                ),

            "n_candidates":
                int(
                    len(g)
                ),
        })

    return pd.DataFrame(
        rows
    )


def add_deltas(
    metrics,
):

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

    out = metrics.merge(
        base,
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    out[
        "delta_ndcg10"
    ] = (
        out["ndcg10"]
        - out["base_ndcg10"]
    )

    out[
        "delta_pairwise_win_rate"
    ] = (
        out[
            "pairwise_win_rate"
        ]
        - out[
            "base_pairwise_win_rate"
        ]
    )

    out[
        "delta_pairwise_margin"
    ] = (
        out[
            "pairwise_margin"
        ]
        - out[
            "base_pairwise_margin"
        ]
    )

    return out


def condition_significance(
    case_table,
    dataset,
    model,
):

    rows = []

    base = (
        case_table[
            case_table[
                "condition"
            ] == "base"
        ]
        .set_index(
            "case_id"
        )
    )

    for condition in PERTURBATIONS:

        x = (
            case_table[
                case_table[
                    "condition"
                ] == condition
            ]
            .set_index(
                "case_id"
            )
        )

        common = (
            base.index
            .intersection(
                x.index
            )
        )

        for metric in [
            "ndcg10",
            "pairwise_win_rate",
        ]:

            diff = (
                x.loc[
                    common,
                    metric,
                ].to_numpy()
                -
                base.loc[
                    common,
                    metric,
                ].to_numpy()
            )

            stats = paired_stats(
                diff,
                SEED
                + len(rows),
            )

            rows.append({
                "dataset":
                    dataset,

                "model":
                    model,

                "condition":
                    condition,

                "metric":
                    metric,

                **stats,
            })

    out = pd.DataFrame(
        rows
    )

    for metric in [
        "ndcg10",
        "pairwise_win_rate",
    ]:

        mask = (
            out[
                "metric"
            ] == metric
        )

        out.loc[
            mask,
            "holm_p"
        ] = holm_adjust(
            out.loc[
                mask,
                "wilcoxon_p"
            ].to_numpy()
        )

    return out


def quality_vs_clean(
    case_table,
    dataset,
    model,
):

    pivot = (
        case_table.pivot(
            index="case_id",
            columns="condition",
            values="delta_ndcg10",
        )
    )

    rows = []

    for condition in [
        "partial",
        "redundant",
        "irrelevant",
        "conflict",
    ]:

        pair = pivot[
            [
                "clean",
                condition,
            ]
        ].dropna()

        diff = (
            pair[
                condition
            ]
            - pair[
                "clean"
            ]
        )

        rows.append({
            "dataset":
                dataset,

            "model":
                model,

            "condition":
                condition,

            "n":
                len(pair),

            "condition_better_than_clean_rate":
                float(
                    (
                        diff > 1e-15
                    ).mean()
                ),

            "condition_worse_than_clean_rate":
                float(
                    (
                        diff < -1e-15
                    ).mean()
                ),

            "equal_to_clean_rate":
                float(
                    (
                        np.abs(diff)
                        <= 1e-15
                    ).mean()
                ),

            "mean_utility_gap_vs_clean":
                float(
                    diff.mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


def knowledge_correlation(
    case_table,
    case_info,
    result_dir,
    dataset,
    model,
):

    p = (
        result_dir
        / "item_knowledge.csv"
    )

    if not p.exists():
        return None

    k = pd.read_csv(p)

    if (
        "item_id"
        not in k.columns
        or
        "knowledge_score"
        not in k.columns
    ):
        return None

    k = (
        k[
            [
                "item_id",
                "knowledge_score",
            ]
        ]
        .copy()
    )

    k[
        "item_id"
    ] = (
        k[
            "item_id"
        ]
        .astype(str)
    )

    k = (
        k.drop_duplicates(
            "item_id"
        )
    )

    clean = (
        case_table[
            case_table[
                "condition"
            ] == "clean"
        ][
            [
                "case_id",
                "delta_ndcg10",
                "delta_pairwise_win_rate",
            ]
        ]
        .merge(
            case_info[
                [
                    "case_id",
                    "target_item_id",
                ]
            ],
            on="case_id",
            how="left",
        )
    )

    clean = clean.merge(
        k,
        left_on="target_item_id",
        right_on="item_id",
        how="left",
    )

    clean = clean.dropna(
        subset=[
            "knowledge_score"
        ]
    )

    if len(clean) < 3:
        return None

    rows = []

    for metric in [
        "delta_ndcg10",
        "delta_pairwise_win_rate",
    ]:

        rho, pvalue = spearmanr(
            clean[
                "knowledge_score"
            ],
            clean[
                metric
            ],
        )

        rows.append({
            "dataset":
                dataset,

            "model":
                model,

            "metric":
                metric,

            "n":
                int(
                    len(clean)
                ),

            "rho":
                float(rho),

            "p":
                float(pvalue),
        })

    return pd.DataFrame(
        rows
    )


def main():

    manifests = sorted(
        BENCH_ROOT.glob(
            "*/*/results/"
            "benchmark_manifest.json"
        )
    )

    if len(manifests) != 6:
        raise RuntimeError(
            f"Expected 6 benchmark runs, "
            f"found {len(manifests)}"
        )

    global_rows = []
    significance_rows = []
    quality_rows = []
    knowledge_rows = []

    case_tables = {}

    metric_shift_rows = []

    print(
        "Discovered runs =",
        len(manifests)
    )

    for manifest_path in manifests:

        result_dir = (
            manifest_path.parent
        )

        model_dir = (
            result_dir.parent
        )

        dataset_dir = (
            model_dir.parent
        )

        dataset = (
            dataset_dir.name
        )

        model = (
            model_dir.name
        )

        data_dir = (
            dataset_dir
            / "data"
        )

        print()
        print(
            "=" * 80
        )
        print(
            dataset,
            "×",
            model,
        )
        print(
            "=" * 80
        )

        case_info = (
            build_case_info(
                data_dir
            )
        )

        metrics = (
            recompute_case_metrics(
                result_dir
            )
        )

        got = set(
            metrics[
                "condition"
            ].unique()
        )

        if got != set(CONDITIONS):

            raise RuntimeError(
                f"{dataset} × {model}: "
                f"conditions={got}"
            )

        counts = (
            metrics.groupby(
                "condition"
            )[
                "case_id"
            ]
            .nunique()
        )

        if (
            counts.nunique()
            != 1
        ):
            raise RuntimeError(
                f"Case count mismatch:\n"
                f"{counts}"
            )

        metrics = metrics.merge(
            case_info,
            on="case_id",
            how="left",
            validate="many_to_one",
        )

        metrics = add_deltas(
            metrics
        )

        case_tables[
            (
                dataset,
                model,
            )
        ] = metrics.copy()

        metrics.to_csv(
            result_dir
            / "case_metrics_tieaware.csv",
            index=False,
        )

        global_summary = summarize(
            metrics,
            [
                "condition",
            ],
        )

        global_summary[
            "dataset"
        ] = dataset

        global_summary[
            "model"
        ] = model

        global_summary.to_csv(
            result_dir
            / "benchmark_summary_tieaware.csv",
            index=False,
        )

        by_pop = summarize(
            metrics,
            [
                "condition",
                "popularity_stratum",
            ],
        )

        by_pop[
            "dataset"
        ] = dataset

        by_pop[
            "model"
        ] = model

        by_pop.to_csv(
            result_dir
            / "benchmark_by_popularity_tieaware.csv",
            index=False,
        )

        global_rows.append(
            global_summary
        )

        sig = (
            condition_significance(
                metrics,
                dataset,
                model,
            )
        )

        significance_rows.append(
            sig
        )

        quality_rows.append(
            quality_vs_clean(
                metrics,
                dataset,
                model,
            )
        )

        kc = knowledge_correlation(
            metrics,
            case_info,
            result_dir,
            dataset,
            model,
        )

        if kc is not None:
            knowledge_rows.append(
                kc
            )

        # Compare old metrics against tie-aware metrics.
        old_path = (
            result_dir
            / "benchmark_summary.csv"
        )

        if old_path.exists():

            old = pd.read_csv(
                old_path
            )[
                [
                    "condition",
                    "ndcg10",
                ]
            ].rename(
                columns={
                    "ndcg10":
                        "old_ndcg10",
                }
            )

            new = global_summary[
                [
                    "condition",
                    "ndcg10",
                ]
            ].rename(
                columns={
                    "ndcg10":
                        "tieaware_ndcg10",
                }
            )

            shift = old.merge(
                new,
                on="condition",
            )

            shift[
                "difference"
            ] = (
                shift[
                    "tieaware_ndcg10"
                ]
                -
                shift[
                    "old_ndcg10"
                ]
            )

            shift[
                "dataset"
            ] = dataset

            shift[
                "model"
            ] = model

            metric_shift_rows.append(
                shift
            )

        print(
            global_summary[
                [
                    "condition",
                    "n",
                    "ndcg10",
                    "delta_ndcg10",
                    "ndcg_harmful_rate",
                    "ndcg_helpful_rate",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ==================================================
    # Global condition matrix
    # ==================================================

    global_df = pd.concat(
        global_rows,
        ignore_index=True,
    )

    global_df.to_csv(
        FINAL_DIR
        / "global_matrix_tieaware.csv",
        index=False,
    )

    # ==================================================
    # Statistical significance
    # ==================================================

    sig_df = pd.concat(
        significance_rows,
        ignore_index=True,
    )

    sig_df.to_csv(
        FINAL_DIR
        / "condition_significance.csv",
        index=False,
    )

    # ==================================================
    # Cross-setting condition consistency
    # ==================================================

    ndcg_sig = sig_df[
        sig_df[
            "metric"
        ] == "ndcg10"
    ].copy()

    consistency = (
        ndcg_sig.groupby(
            "condition"
        )
        .agg(
            settings=(
                "dataset",
                "count",
            ),

            mean_delta=(
                "mean_delta",
                "mean",
            ),

            median_delta=(
                "mean_delta",
                "median",
            ),

            min_delta=(
                "mean_delta",
                "min",
            ),

            max_delta=(
                "mean_delta",
                "max",
            ),

            positive_settings=(
                "mean_delta",
                lambda x:
                    int(
                        (
                            x > 0
                        ).sum()
                    ),
            ),

            negative_settings=(
                "mean_delta",
                lambda x:
                    int(
                        (
                            x < 0
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    significant_positive = (
        ndcg_sig[
            (
                ndcg_sig[
                    "holm_p"
                ] < 0.05
            )
            &
            (
                ndcg_sig[
                    "mean_delta"
                ] > 0
            )
        ]
        .groupby(
            "condition"
        )
        .size()
    )

    significant_negative = (
        ndcg_sig[
            (
                ndcg_sig[
                    "holm_p"
                ] < 0.05
            )
            &
            (
                ndcg_sig[
                    "mean_delta"
                ] < 0
            )
        ]
        .groupby(
            "condition"
        )
        .size()
    )

    consistency[
        "significant_positive_settings"
    ] = (
        consistency[
            "condition"
        ]
        .map(
            significant_positive
        )
        .fillna(0)
        .astype(int)
    )

    consistency[
        "significant_negative_settings"
    ] = (
        consistency[
            "condition"
        ]
        .map(
            significant_negative
        )
        .fillna(0)
        .astype(int)
    )

    consistency.to_csv(
        FINAL_DIR
        / "condition_consistency.csv",
        index=False,
    )

    # ==================================================
    # Quality mismatch
    # ==================================================

    quality_df = pd.concat(
        quality_rows,
        ignore_index=True,
    )

    quality_df.to_csv(
        FINAL_DIR
        / "quality_vs_clean.csv",
        index=False,
    )

    # ==================================================
    # Knowledge-gap correlation
    # ==================================================

    if knowledge_rows:

        knowledge_df = pd.concat(
            knowledge_rows,
            ignore_index=True,
        )

        knowledge_df.to_csv(
            FINAL_DIR
            / "knowledge_utility_correlation.csv",
            index=False,
        )

    # ==================================================
    # Model polarity flips: 3B vs 7B
    # ==================================================

    flip_rows = []

    datasets = sorted(
        {
            x[0]
            for x in case_tables
        }
    )

    for dataset in datasets:

        key3 = (
            dataset,
            "Qwen2.5-3B-Instruct",
        )

        key7 = (
            dataset,
            "Qwen2.5-7B-Instruct",
        )

        if (
            key3 not in case_tables
            or key7 not in case_tables
        ):
            continue

        d3 = case_tables[
            key3
        ]

        d7 = case_tables[
            key7
        ]

        for condition in PERTURBATIONS:

            a = (
                d3[
                    d3[
                        "condition"
                    ] == condition
                ][
                    [
                        "case_id",
                        "delta_ndcg10",
                    ]
                ]
                .rename(
                    columns={
                        "delta_ndcg10":
                            "delta_3b",
                    }
                )
            )

            b = (
                d7[
                    d7[
                        "condition"
                    ] == condition
                ][
                    [
                        "case_id",
                        "delta_ndcg10",
                    ]
                ]
                .rename(
                    columns={
                        "delta_ndcg10":
                            "delta_7b",
                    }
                )
            )

            x = a.merge(
                b,
                on="case_id",
                how="inner",
                validate="one_to_one",
            )

            pos3_neg7 = (
                (
                    x["delta_3b"] > 1e-15
                )
                &
                (
                    x["delta_7b"] < -1e-15
                )
            )

            neg3_pos7 = (
                (
                    x["delta_3b"] < -1e-15
                )
                &
                (
                    x["delta_7b"] > 1e-15
                )
            )

            both_pos = (
                (
                    x["delta_3b"] > 1e-15
                )
                &
                (
                    x["delta_7b"] > 1e-15
                )
            )

            both_neg = (
                (
                    x["delta_3b"] < -1e-15
                )
                &
                (
                    x["delta_7b"] < -1e-15
                )
            )

            rho, pvalue = spearmanr(
                x[
                    "delta_3b"
                ],
                x[
                    "delta_7b"
                ],
            )

            flip_rows.append({
                "dataset":
                    dataset,

                "condition":
                    condition,

                "n":
                    len(x),

                "mean_delta_3b":
                    float(
                        x[
                            "delta_3b"
                        ].mean()
                    ),

                "mean_delta_7b":
                    float(
                        x[
                            "delta_7b"
                        ].mean()
                    ),

                "positive_3b_negative_7b_rate":
                    float(
                        pos3_neg7.mean()
                    ),

                "negative_3b_positive_7b_rate":
                    float(
                        neg3_pos7.mean()
                    ),

                "strict_polarity_flip_rate":
                    float(
                        (
                            pos3_neg7
                            |
                            neg3_pos7
                        ).mean()
                    ),

                "both_positive_rate":
                    float(
                        both_pos.mean()
                    ),

                "both_negative_rate":
                    float(
                        both_neg.mean()
                    ),

                "utility_spearman_3b_7b":
                    float(rho),

                "utility_spearman_p":
                    float(pvalue),
            })

    flip_df = pd.DataFrame(
        flip_rows
    )

    flip_df.to_csv(
        FINAL_DIR
        / "model_polarity_flip.csv",
        index=False,
    )

    # ==================================================
    # Old vs tie-aware shift
    # ==================================================

    if metric_shift_rows:

        shifts = pd.concat(
            metric_shift_rows,
            ignore_index=True,
        )

        shifts.to_csv(
            FINAL_DIR
            / "tieaware_metric_shift.csv",
            index=False,
        )

        print()
        print(
            "Max |tie-aware - old NDCG| =",
            shifts[
                "difference"
            ].abs().max()
        )

    print()
    print(
        "=" * 80
    )
    print(
        "FINAL BENCHMARK FILES SAVED TO:"
    )
    print(
        FINAL_DIR
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
