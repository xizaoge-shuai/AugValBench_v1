import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.utility.train_ranker import FEATURES
from src.utility.evaluate_ranker import expected_tie_metrics


SEED = 2027

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

OUT = (
    config.RESULT_DIR
    / "method_listwise_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Model
# =========================================================

def make_regressor():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    ])


# =========================================================
# Listwise features
#
# IMPORTANT:
# All features below are computed ONLY from pre-augmentation
# base scores. No clean_score, target label, or post-treatment
# information is used.
# =========================================================

def add_listwise_features(df):

    out = df.copy()

    parts = []

    for case_id, g in out.groupby(
        "case_id",
        sort=False,
    ):

        x = g.copy()

        scores = (
            x["base_score"]
            .to_numpy(
                dtype=float
            )
        )

        n = len(scores)

        order = np.argsort(
            -scores,
            kind="stable",
        )

        sorted_scores = (
            scores[order]
        )

        rank = (
            pd.Series(scores)
            .rank(
                method="average",
                ascending=False,
            )
            .to_numpy(
                dtype=float
            )
        )

        case_max = float(
            np.max(scores)
        )

        case_min = float(
            np.min(scores)
        )

        case_median = float(
            np.median(scores)
        )

        case_q25 = float(
            np.quantile(
                scores,
                0.25,
            )
        )

        case_q75 = float(
            np.quantile(
                scores,
                0.75,
            )
        )

        case_range = (
            case_max
            - case_min
        )

        case_iqr = (
            case_q75
            - case_q25
        )

        # -------------------------------------------------
        # Neighbor gaps
        # -------------------------------------------------

        gap_above = np.zeros(
            n,
            dtype=float,
        )

        gap_below = np.zeros(
            n,
            dtype=float,
        )

        for pos, idx in enumerate(
            order
        ):

            if pos == 0:

                gap_above[idx] = (
                    case_range
                )

            else:

                gap_above[idx] = (
                    sorted_scores[
                        pos - 1
                    ]
                    -
                    sorted_scores[
                        pos
                    ]
                )

            if pos == n - 1:

                gap_below[idx] = (
                    case_range
                )

            else:

                gap_below[idx] = (
                    sorted_scores[
                        pos
                    ]
                    -
                    sorted_scores[
                        pos + 1
                    ]
                )

        nearest_gap = np.minimum(
            gap_above,
            gap_below,
        )

        # -------------------------------------------------
        # Ranking boundaries
        # -------------------------------------------------

        def boundary_score(k):

            if n == 0:
                return 0.0

            idx = min(
                k - 1,
                n - 1,
            )

            return float(
                sorted_scores[idx]
            )

        top1_boundary = (
            boundary_score(1)
        )

        top3_boundary = (
            boundary_score(3)
        )

        top5_boundary = (
            boundary_score(5)
        )

        top10_boundary = (
            boundary_score(10)
        )

        # -------------------------------------------------
        # Local competition density
        # -------------------------------------------------

        diff_matrix = np.abs(
            scores[:, None]
            -
            scores[None, :]
        )

        close_001 = (
            (
                diff_matrix
                <= 0.01
            ).sum(
                axis=1
            )
            - 1
        )

        close_005 = (
            (
                diff_matrix
                <= 0.05
            ).sum(
                axis=1
            )
            - 1
        )

        close_010 = (
            (
                diff_matrix
                <= 0.10
            ).sum(
                axis=1
            )
            - 1
        )

        # -------------------------------------------------
        # Softmax competition
        # -------------------------------------------------

        shifted = (
            scores
            - np.max(scores)
        )

        exp_scores = np.exp(
            shifted
        )

        softmax = (
            exp_scores
            /
            np.sum(
                exp_scores
            )
        )

        entropy = float(
            -np.sum(
                softmax
                * np.log(
                    softmax
                    + 1e-12
                )
            )
        )

        # -------------------------------------------------
        # Candidate-level list-conditioned features
        # -------------------------------------------------

        x[
            "list_rank"
        ] = rank

        x[
            "list_rank_norm"
        ] = (
            (rank - 1.0)
            /
            max(
                n - 1,
                1,
            )
        )

        x[
            "list_reciprocal_rank"
        ] = (
            1.0
            / rank
        )

        x[
            "list_gap_to_max"
        ] = (
            case_max
            - scores
        )

        x[
            "list_gap_to_min"
        ] = (
            scores
            - case_min
        )

        x[
            "list_gap_to_median"
        ] = (
            scores
            - case_median
        )

        x[
            "list_gap_above"
        ] = gap_above

        x[
            "list_gap_below"
        ] = gap_below

        x[
            "list_nearest_gap"
        ] = nearest_gap

        x[
            "list_top1_boundary_delta"
        ] = (
            scores
            - top1_boundary
        )

        x[
            "list_top3_boundary_delta"
        ] = (
            scores
            - top3_boundary
        )

        x[
            "list_top5_boundary_delta"
        ] = (
            scores
            - top5_boundary
        )

        x[
            "list_top10_boundary_delta"
        ] = (
            scores
            - top10_boundary
        )

        x[
            "list_is_top1"
        ] = (
            rank <= 1
        ).astype(int)

        x[
            "list_is_top3"
        ] = (
            rank <= 3
        ).astype(int)

        x[
            "list_is_top5"
        ] = (
            rank <= 5
        ).astype(int)

        x[
            "list_is_top10"
        ] = (
            rank <= 10
        ).astype(int)

        x[
            "list_case_min"
        ] = case_min

        x[
            "list_case_max"
        ] = case_max

        x[
            "list_case_median"
        ] = case_median

        x[
            "list_case_q25"
        ] = case_q25

        x[
            "list_case_q75"
        ] = case_q75

        x[
            "list_case_range"
        ] = case_range

        x[
            "list_case_iqr"
        ] = case_iqr

        x[
            "list_close_001"
        ] = (
            close_001
            / max(
                n - 1,
                1,
            )
        )

        x[
            "list_close_005"
        ] = (
            close_005
            / max(
                n - 1,
                1,
            )
        )

        x[
            "list_close_010"
        ] = (
            close_010
            / max(
                n - 1,
                1,
            )
        )

        x[
            "list_softmax_share"
        ] = softmax

        x[
            "list_entropy"
        ] = entropy

        parts.append(x)

    return pd.concat(
        parts,
        ignore_index=True,
    )


LIST_FEATURES = [
    "list_rank",
    "list_rank_norm",
    "list_reciprocal_rank",

    "list_gap_to_max",
    "list_gap_to_min",
    "list_gap_to_median",

    "list_gap_above",
    "list_gap_below",
    "list_nearest_gap",

    "list_top1_boundary_delta",
    "list_top3_boundary_delta",
    "list_top5_boundary_delta",
    "list_top10_boundary_delta",

    "list_is_top1",
    "list_is_top3",
    "list_is_top5",
    "list_is_top10",

    "list_case_min",
    "list_case_max",
    "list_case_median",
    "list_case_q25",
    "list_case_q75",
    "list_case_range",
    "list_case_iqr",

    "list_close_001",
    "list_close_005",
    "list_close_010",

    "list_softmax_share",
    "list_entropy",
]


# =========================================================
# Ranking evaluation
# =========================================================

def evaluate(
    frame,
    prediction,
    budget,
    abstain=False,
):

    work = frame.copy()

    work[
        "_prediction"
    ] = np.asarray(
        prediction,
        dtype=float,
    )

    rows = []

    for case_id, g in work.groupby(
        "case_id",
        sort=False,
    ):

        g = g.reset_index(
            drop=True
        )

        pred = (
            g[
                "_prediction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        n = len(g)

        k = max(
            1,
            int(
                round(
                    budget * n
                )
            ),
        )

        order = np.argsort(
            -pred,
            kind="stable",
        )

        if abstain:

            order = order[
                pred[
                    order
                ] > 0.0
            ]

        selected = (
            order[:k]
        )

        base_scores = (
            g[
                "base_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        clean_scores = (
            g[
                "clean_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        labels = (
            g[
                "is_target"
            ]
            .to_numpy(
                dtype=int
            )
        )

        final_scores = (
            base_scores.copy()
        )

        final_scores[
            selected
        ] = (
            clean_scores[
                selected
            ]
        )

        metrics = (
            expected_tie_metrics(
                final_scores,
                labels,
            )
        )

        true_utility = (
            g[
                "utility_ndcg_if_aug"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(selected):

            selected_u = (
                true_utility[
                    selected
                ]
            )

            harmful = float(
                (
                    selected_u < 0
                ).mean()
            )

            helpful = float(
                (
                    selected_u > 0
                ).mean()
            )

        else:

            harmful = 0.0
            helpful = 0.0

        rows.append({
            "case_id":
                int(case_id),

            "ndcg10":
                metrics[
                    "ndcg10"
                ],

            "hr10":
                metrics[
                    "hr10"
                ],

            "mrr":
                metrics[
                    "mrr"
                ],

            "pairwise_win_rate":
                metrics[
                    "pairwise_win_rate"
                ],

            "augment_ratio":
                len(selected)
                / n,

            "selected_harm_rate":
                harmful,

            "selected_help_rate":
                helpful,
        })

    return pd.DataFrame(
        rows
    )


def summarize(
    cases,
):

    return {
        "ndcg10":
            float(
                cases[
                    "ndcg10"
                ].mean()
            ),

        "hr10":
            float(
                cases[
                    "hr10"
                ].mean()
            ),

        "mrr":
            float(
                cases[
                    "mrr"
                ].mean()
            ),

        "pairwise_win_rate":
            float(
                cases[
                    "pairwise_win_rate"
                ].mean()
            ),

        "augment_ratio":
            float(
                cases[
                    "augment_ratio"
                ].mean()
            ),

        "selected_harm_rate":
            float(
                cases[
                    "selected_harm_rate"
                ].mean()
            ),

        "selected_help_rate":
            float(
                cases[
                    "selected_help_rate"
                ].mean()
            ),
    }


def run_router(
    train,
    val,
    features,
    router,
    abstain,
):

    model = make_regressor()

    model.fit(
        train[
            features
        ],
        train[
            "utility_ndcg_if_aug"
        ],
    )

    pred = model.predict(
        val[
            features
        ]
    )

    rows = []

    for budget in BUDGETS:

        cases = evaluate(
            val,
            pred,
            budget,
            abstain=abstain,
        )

        s = summarize(
            cases
        )

        rows.append({
            "router":
                router,

            "budget":
                budget,

            **s,
        })

    return (
        pd.DataFrame(rows),
        pred,
    )


def main():

    benefit_path = (
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )

    split_path = (
        config.DATA_DIR
        / "formal"
        / "split_manifest.csv"
    )

    if not benefit_path.exists():

        raise FileNotFoundError(
            benefit_path
        )

    if not split_path.exists():

        raise FileNotFoundError(
            split_path
        )

    df = pd.read_csv(
        benefit_path
    )

    split = pd.read_csv(
        split_path
    )

    df[
        "user_id"
    ] = (
        df[
            "user_id"
        ]
        .astype(str)
    )

    split[
        "user_id"
    ] = (
        split[
            "user_id"
        ]
        .astype(str)
    )

    df = df.merge(
        split,
        on=[
            "case_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if df[
        "split"
    ].isna().any():

        raise RuntimeError(
            "Missing split labels."
        )

    required = {
        "case_id",
        "user_id",
        "base_score",
        "clean_score",
        "is_target",
        "utility_ndcg_if_aug",
        "split",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # -----------------------------------------------------
    # Add listwise pre-treatment features.
    # -----------------------------------------------------

    df = add_listwise_features(
        df
    )

    base_features = [
        f
        for f in FEATURES
        if f in df.columns
    ]

    listwise_features = (
        base_features
        + LIST_FEATURES
    )

    forbidden = {
        "is_target",
        "clean_score",
        "benefit_label",
        "utility_ndcg_if_aug",
        "utility_pwr_if_aug",
        "utility_margin_if_aug",
    }

    leakage = (
        set(
            listwise_features
        )
        & forbidden
    )

    if leakage:

        raise RuntimeError(
            f"LEAKAGE DETECTED: "
            f"{leakage}"
        )

    train = (
        df[
            df[
                "split"
            ] == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val = (
        df[
            df[
                "split"
            ] == "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "============================================"
    )

    print(
        "dataset =",
        config.DATASET_NAME
    )

    print(
        "model =",
        config.MODEL_NAME
    )

    print(
        "train cases =",
        train[
            "case_id"
        ].nunique()
    )

    print(
        "val cases =",
        val[
            "case_id"
        ].nunique()
    )

    print(
        "base features =",
        len(
            base_features
        )
    )

    print(
        "listwise features =",
        len(
            LIST_FEATURES
        )
    )

    print(
        "total features =",
        len(
            listwise_features
        )
    )

    print(
        "============================================"
    )

    # =====================================================
    # A. Independent baseline reproduction
    # =====================================================

    independent_curve, independent_pred = (
        run_router(
            train,
            val,
            base_features,
            "IndependentRF",
            abstain=False,
        )
    )

    # =====================================================
    # B. List-conditioned exact-budget
    # =====================================================

    listwise_curve, listwise_pred = (
        run_router(
            train,
            val,
            listwise_features,
            "ListwiseRF",
            abstain=False,
        )
    )

    # =====================================================
    # C. List-conditioned + abstention
    # =====================================================

    listwise_abstain_curve, _ = (
        run_router(
            train,
            val,
            listwise_features,
            "ListwiseRFAbstain",
            abstain=True,
        )
    )

    curve = pd.concat(
        [
            independent_curve,
            listwise_curve,
            listwise_abstain_curve,
        ],
        ignore_index=True,
    )

    curve.to_csv(
        OUT
        / "validation_budget_curve.csv",
        index=False,
    )

    selected_rows = []

    for router, g in curve.groupby(
        "router",
        sort=False,
    ):

        best = (
            g.sort_values(
                [
                    "ndcg10",
                    "augment_ratio",
                    "budget",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        selected_rows.append(
            best.to_dict()
        )

    selected = pd.DataFrame(
        selected_rows
    )

    selected[
        "dataset"
    ] = config.DATASET_NAME

    selected[
        "model"
    ] = config.MODEL_NAME

    selected.to_csv(
        OUT
        / "validation_selected.csv",
        index=False,
    )

    pred_out = val[
        [
            "case_id",
            "user_id",
        ]
    ].copy()

    pred_out[
        "true_utility"
    ] = (
        val[
            "utility_ndcg_if_aug"
        ].to_numpy()
    )

    pred_out[
        "independent_pred"
    ] = independent_pred

    pred_out[
        "listwise_pred"
    ] = listwise_pred

    pred_out.to_csv(
        OUT
        / "validation_predictions.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

        "seed":
            SEED,

        "budgets":
            BUDGETS,

        "base_features":
            base_features,

        "list_features":
            LIST_FEATURES,

        "total_features":
            listwise_features,

        "test_evaluated":
            False,
    }

    with (
        OUT
        / "manifest.json"
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

    print()
    print(
        "===== VALIDATION SELECTED ====="
    )

    print(
        selected[
            [
                "router",
                "budget",
                "ndcg10",
                "hr10",
                "mrr",
                "pairwise_win_rate",
                "augment_ratio",
                "selected_harm_rate",
                "selected_help_rate",
            ]
        ]
        .sort_values(
            "ndcg10",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print()
    print(
        "Saved:",
        OUT
    )


if __name__ == "__main__":
    main()
