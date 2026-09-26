import json
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.utility.train_ranker import FEATURES
from src.utility.evaluate_ranker import (
    expected_tie_metrics,
)


OUT = (
    config.RESULT_DIR
    / f"formal_{config.DATASET_NAME.lower()}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

VAL_RANDOM_REPEATS = 30
TEST_RANDOM_REPEATS = 100

SEED = 2027


def make_regressor(seed):
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
                random_state=seed,
            ),
        ),
    ])


def make_classifier(seed):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ])


def positive_probability(
    model,
    X,
):
    probs = model.predict_proba(
        X
    )

    classes = (
        model.named_steps[
            "model"
        ].classes_
    )

    if 1 not in classes:
        return np.zeros(
            len(X),
            dtype=float,
        )

    idx = int(
        np.where(
            classes == 1
        )[0][0]
    )

    return probs[:, idx]


def topk(values, k):
    values = np.asarray(
        values,
        dtype=float,
    )

    return np.argsort(
        -values,
        kind="stable",
    )[:k]


def evaluate_with_scores(
    frame,
    selector_scores,
    budget,
    router,
):

    work = frame.copy()

    work[
        "_selector_score"
    ] = np.asarray(
        selector_scores,
        dtype=float,
    )

    rows = []

    for case_id, case in work.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        n = len(case)

        if budget <= 0:
            selected = np.array(
                [],
                dtype=int,
            )

        elif budget >= 1:
            selected = np.arange(
                n,
                dtype=int,
            )

        else:
            k = max(
                1,
                int(
                    round(
                        budget * n
                    )
                ),
            )

            selected = topk(
                case[
                    "_selector_score"
                ].to_numpy(),
                k,
            )

        base = (
            case[
                "base_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        clean = (
            case[
                "clean_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        labels = (
            case[
                "is_target"
            ]
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

        m = expected_tie_metrics(
            score,
            labels,
        )

        rows.append({
            "case_id":
                int(case_id),
            "router":
                router,
            "budget":
                float(budget),
            "augment_ratio":
                float(
                    len(selected)
                    / n
                ),
            **m,
        })

    return pd.DataFrame(
        rows
    )


def evaluate_random(
    frame,
    budget,
    repeats,
    seed,
    router="Random",
):

    rows = []

    for case_id, case in frame.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        n = len(case)

        if budget <= 0:
            k = 0

        elif budget >= 1:
            k = n

        else:
            k = max(
                1,
                int(
                    round(
                        budget * n
                    )
                ),
            )

        rep_metrics = []

        for rep in range(
            repeats
        ):
            rng = random.Random(
                seed
                + int(case_id) * 1009
                + rep * 97
                + int(
                    budget * 1000
                ) * 17
            )

            ids = list(
                range(n)
            )

            rng.shuffle(ids)

            selected = np.asarray(
                ids[:k],
                dtype=int,
            )

            base = (
                case[
                    "base_score"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            clean = (
                case[
                    "clean_score"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            labels = (
                case[
                    "is_target"
                ]
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

            rep_metrics.append(
                expected_tie_metrics(
                    score,
                    labels,
                )
            )

        avg = {
            key:
                float(
                    np.mean(
                        [
                            x[key]
                            for x
                            in rep_metrics
                        ]
                    )
                )
            for key
            in rep_metrics[0]
        }

        rows.append({
            "case_id":
                int(case_id),
            "router":
                router,
            "budget":
                float(budget),
            "augment_ratio":
                float(
                    k / n
                ),
            **avg,
        })

    return pd.DataFrame(
        rows
    )


def mean_ndcg(case_df):
    return float(
        case_df[
            "ndcg10"
        ].mean()
    )


def select_budget(
    frame,
    scores,
    router,
):

    records = []

    for budget in BUDGETS:

        case_df = (
            evaluate_with_scores(
                frame,
                scores,
                budget,
                router,
            )
        )

        records.append({
            "router":
                router,
            "budget":
                budget,
            "val_ndcg10":
                mean_ndcg(
                    case_df
                ),
            "val_hr10":
                float(
                    case_df[
                        "hr10"
                    ].mean()
                ),
            "val_pwr":
                float(
                    case_df[
                        "pairwise_win_rate"
                    ].mean()
                ),
        })

    table = pd.DataFrame(
        records
    )

    # Highest validation NDCG.
    # On exact ties prefer lower augmentation cost.
    best = (
        table.sort_values(
            [
                "val_ndcg10",
                "budget",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(
            best["budget"]
        ),
        table,
    )


def select_random_budget(
    frame,
):

    records = []

    for budget in BUDGETS:

        case_df = evaluate_random(
            frame,
            budget,
            VAL_RANDOM_REPEATS,
            SEED,
        )

        records.append({
            "router":
                "Random",
            "budget":
                budget,
            "val_ndcg10":
                float(
                    case_df[
                        "ndcg10"
                    ].mean()
                ),
            "val_hr10":
                float(
                    case_df[
                        "hr10"
                    ].mean()
                ),
            "val_pwr":
                float(
                    case_df[
                        "pairwise_win_rate"
                    ].mean()
                ),
        })

    table = pd.DataFrame(
        records
    )

    best = (
        table.sort_values(
            [
                "val_ndcg10",
                "budget",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(
            best["budget"]
        ),
        table,
    )


def main():

    data_path = (
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )

    split_path = (
        config.DATA_DIR
        / "formal"
        / "split_manifest.csv"
    )

    df = pd.read_csv(
        data_path
    )

    split = pd.read_csv(
        split_path
    )

    df[
        "user_id"
    ] = df[
        "user_id"
    ].astype(str)

    split[
        "user_id"
    ] = split[
        "user_id"
    ].astype(str)

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
            "Some cases are missing split labels."
        )

    features = [
        f
        for f in FEATURES
        if f in df.columns
    ]

    forbidden = {
        "is_target",
        "clean_score",
        "benefit_label",
        "utility_ndcg_if_aug",
        "utility_pwr_if_aug",
        "utility_margin_if_aug",
    }

    leakage = (
        set(features)
        & forbidden
    )

    if leakage:
        raise RuntimeError(
            f"Feature leakage: {leakage}"
        )

    train = (
        df[
            df["split"]
            == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val = (
        df[
            df["split"]
            == "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        df[
            df["split"]
            == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "\n===== FORMAL DATA ====="
    )

    for name, x in [
        (
            "train",
            train,
        ),
        (
            "val",
            val,
        ),
        (
            "test",
            test,
        ),
    ]:

        print(
            f"{name:5s}: "
            f"cases="
            f"{x['case_id'].nunique()}, "
            f"rows={len(x)}"
        )

    print(
        "features =",
        len(features)
    )

    X_train = train[
        features
    ]

    X_val = val[
        features
    ]

    X_test = test[
        features
    ]

    # ==================================================
    # 1. DirectNDCG
    # ==================================================

    direct = make_regressor(
        SEED
    )

    direct.fit(
        X_train,
        train[
            "utility_ndcg_if_aug"
        ],
    )

    direct_val_score = (
        direct.predict(
            X_val
        )
    )

    direct_budget, direct_search = (
        select_budget(
            val,
            direct_val_score,
            "DirectNDCG",
        )
    )

    # ==================================================
    # 2. NDCG-sign classifier baseline
    # ==================================================

    sign_ndcg = make_classifier(
        SEED + 100
    )

    y_sign_ndcg = (
        train[
            "utility_ndcg_if_aug"
        ]
        > 1e-12
    ).astype(int)

    sign_ndcg.fit(
        X_train,
        y_sign_ndcg,
    )

    sign_ndcg_val = (
        positive_probability(
            sign_ndcg,
            X_val,
        )
    )

    sign_ndcg_budget, sign_ndcg_search = (
        select_budget(
            val,
            sign_ndcg_val,
            "BenefitRF_NDCG",
        )
    )

    # ==================================================
    # 3. Existing margin-sign classifier baseline
    # ==================================================

    sign_margin = make_classifier(
        SEED + 200
    )

    sign_margin.fit(
        X_train,
        train[
            "benefit_label"
        ].astype(int),
    )

    sign_margin_val = (
        positive_probability(
            sign_margin,
            X_val,
        )
    )

    sign_margin_budget, sign_margin_search = (
        select_budget(
            val,
            sign_margin_val,
            "BenefitRF_Margin",
        )
    )

    # ==================================================
    # 4. Heuristic baselines
    # ==================================================

    val_scores = {
        "Tail":
            -val[
                "item_popularity_log"
            ].to_numpy(),

        "LowKnowledge":
            -val[
                "knowledge_score"
            ]
            .fillna(1.0)
            .to_numpy(),

        "Uncertainty":
            -np.abs(
                val[
                    "base_score"
                ].to_numpy()
                - 0.5
            ),

        "OracleNDCG":
            val[
                "utility_ndcg_if_aug"
            ].to_numpy(),
    }

    selected_budgets = {
        "DirectNDCG":
            direct_budget,

        "BenefitRF_NDCG":
            sign_ndcg_budget,

        "BenefitRF_Margin":
            sign_margin_budget,
    }

    search_tables = [
        direct_search,
        sign_ndcg_search,
        sign_margin_search,
    ]

    for router, score in (
        val_scores.items()
    ):

        budget, table = (
            select_budget(
                val,
                score,
                router,
            )
        )

        selected_budgets[
            router
        ] = budget

        search_tables.append(
            table
        )

    random_budget, random_table = (
        select_random_budget(
            val
        )
    )

    selected_budgets[
        "Random"
    ] = random_budget

    search_tables.append(
        random_table
    )

    val_search = pd.concat(
        search_tables,
        ignore_index=True,
    )

    val_search.to_csv(
        OUT
        / "validation_budget_search.csv",
        index=False,
    )

    # ==================================================
    # Refit learned policies on train + validation
    # ==================================================

    train_val = pd.concat(
        [
            train,
            val,
        ],
        ignore_index=True,
    )

    X_train_val = train_val[
        features
    ]

    direct_final = make_regressor(
        SEED
    )

    direct_final.fit(
        X_train_val,
        train_val[
            "utility_ndcg_if_aug"
        ],
    )

    direct_test = (
        direct_final.predict(
            X_test
        )
    )

    sign_ndcg_final = make_classifier(
        SEED + 100
    )

    sign_ndcg_final.fit(
        X_train_val,
        (
            train_val[
                "utility_ndcg_if_aug"
            ]
            > 1e-12
        ).astype(int),
    )

    sign_ndcg_test = (
        positive_probability(
            sign_ndcg_final,
            X_test,
        )
    )

    sign_margin_final = make_classifier(
        SEED + 200
    )

    sign_margin_final.fit(
        X_train_val,
        train_val[
            "benefit_label"
        ].astype(int),
    )

    sign_margin_test = (
        positive_probability(
            sign_margin_final,
            X_test,
        )
    )

    test_scores = {
        "DirectNDCG":
            direct_test,

        "BenefitRF_NDCG":
            sign_ndcg_test,

        "BenefitRF_Margin":
            sign_margin_test,

        "Tail":
            -test[
                "item_popularity_log"
            ].to_numpy(),

        "LowKnowledge":
            -test[
                "knowledge_score"
            ]
            .fillna(1.0)
            .to_numpy(),

        "Uncertainty":
            -np.abs(
                test[
                    "base_score"
                ].to_numpy()
                - 0.5
            ),

        "OracleNDCG":
            test[
                "utility_ndcg_if_aug"
            ].to_numpy(),
    }

    # ==================================================
    # Full fixed-budget TEST curves
    # ==================================================

    curve_parts = []

    for router, score in (
        test_scores.items()
    ):

        for budget in BUDGETS:

            curve_parts.append(
                evaluate_with_scores(
                    test,
                    score,
                    budget,
                    router,
                )
            )

    for budget in BUDGETS:

        curve_parts.append(
            evaluate_random(
                test,
                budget,
                TEST_RANDOM_REPEATS,
                SEED + 500,
                "Random",
            )
        )

    noaug = evaluate_with_scores(
        test,
        np.zeros(
            len(test)
        ),
        0.0,
        "NoAugment",
    )

    uniform = evaluate_with_scores(
        test,
        np.zeros(
            len(test)
        ),
        1.0,
        "UniformClean",
    )

    curve_parts.extend(
        [
            noaug,
            uniform,
        ]
    )

    test_curve_cases = pd.concat(
        curve_parts,
        ignore_index=True,
    )

    test_curve_cases.to_csv(
        OUT
        / "test_budget_case_metrics.csv",
        index=False,
    )

    test_curve_summary = (
        test_curve_cases.groupby(
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

    test_curve_summary.to_csv(
        OUT
        / "test_budget_curve.csv",
        index=False,
    )

    # ==================================================
    # TEST results at VALIDATION-selected budgets
    # ==================================================

    selected_parts = []

    for router, budget in (
        selected_budgets.items()
    ):

        x = test_curve_cases[
            (
                test_curve_cases[
                    "router"
                ] == router
            )
            &
            (
                np.isclose(
                    test_curve_cases[
                        "budget"
                    ],
                    budget,
                )
            )
        ].copy()

        selected_parts.append(
            x
        )

    selected_parts.append(
        noaug
    )

    selected_parts.append(
        uniform
    )

    selected_cases = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected_cases.to_csv(
        OUT
        / "test_selected_case_metrics.csv",
        index=False,
    )

    selected_summary = (
        selected_cases.groupby(
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
        .sort_values(
            "ndcg10",
            ascending=False,
        )
    )

    selected_summary.to_csv(
        OUT
        / "test_selected_summary.csv",
        index=False,
    )

    selection = {
        "seed":
            SEED,

        "train_cases":
            int(
                train[
                    "case_id"
                ].nunique()
            ),

        "val_cases":
            int(
                val[
                    "case_id"
                ].nunique()
            ),

        "test_cases":
            int(
                test[
                    "case_id"
                ].nunique()
            ),

        "features":
            features,

        "regressor": {
            "type":
                "RandomForestRegressor",
            "n_estimators":
                600,
            "max_depth":
                12,
            "min_samples_leaf":
                3,
        },

        "selected_budgets":
            selected_budgets,
    }

    with (
        OUT
        / "selection.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n===== VALIDATION-SELECTED BUDGETS ====="
    )

    for router, budget in (
        selected_budgets.items()
    ):
        print(
            f"{router:20s} "
            f"{budget:.2f}"
        )

    print(
        "\n===== FINAL HELD-OUT TEST ====="
    )

    print(
        selected_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:",
        OUT
    )


if __name__ == "__main__":
    main()
