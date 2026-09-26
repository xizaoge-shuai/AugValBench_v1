import json

import numpy as np
import pandas as pd

from src import config
from src.utility.train_ranker import FEATURES

from src.formal.train_eval_direct_ndcg import (
    BUDGETS,
    SEED,
    TEST_RANDOM_REPEATS,
    make_classifier,
    positive_probability,
    select_budget,
    select_random_budget,
    evaluate_with_scores,
    evaluate_random,
)


OUT = (
    config.RESULT_DIR
    / f"baseline_{config.DATASET_NAME.lower()}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    dataset_path = (
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
        dataset_path
    )

    split = pd.read_csv(
        split_path
    )

    df["user_id"] = (
        df["user_id"]
        .astype(str)
    )

    split["user_id"] = (
        split["user_id"]
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

    if df["split"].isna().any():
        raise RuntimeError(
            "Missing train/val/test split."
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
        .reset_index(drop=True)
    )

    val = (
        df[
            df["split"]
            == "val"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            df["split"]
            == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "===== BASELINE DATA ====="
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
        "train / val / test =",
        train["case_id"].nunique(),
        val["case_id"].nunique(),
        test["case_id"].nunique(),
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
    # Learned baseline 1:
    # positive ΔNDCG classification
    # ==================================================

    clf_ndcg = make_classifier(
        SEED + 100
    )

    clf_ndcg.fit(
        X_train,
        (
            train[
                "utility_ndcg_if_aug"
            ]
            > 1e-12
        ).astype(int),
    )

    val_ndcg_score = (
        positive_probability(
            clf_ndcg,
            X_val,
        )
    )

    (
        ndcg_budget,
        ndcg_search,
    ) = select_budget(
        val,
        val_ndcg_score,
        "BenefitRF_NDCG",
    )

    # ==================================================
    # Learned baseline 2:
    # positive margin classification
    # ==================================================

    clf_margin = make_classifier(
        SEED + 200
    )

    clf_margin.fit(
        X_train,
        train[
            "benefit_label"
        ].astype(int),
    )

    val_margin_score = (
        positive_probability(
            clf_margin,
            X_val,
        )
    )

    (
        margin_budget,
        margin_search,
    ) = select_budget(
        val,
        val_margin_score,
        "BenefitRF_Margin",
    )

    # ==================================================
    # Heuristic baselines
    # ==================================================

    val_selectors = {
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

        # Diagnostic upper bound only.
        "OracleNDCG":
            val[
                "utility_ndcg_if_aug"
            ].to_numpy(),
    }

    selected_budgets = {
        "BenefitRF_NDCG":
            ndcg_budget,

        "BenefitRF_Margin":
            margin_budget,
    }

    search_tables = [
        ndcg_search,
        margin_search,
    ]

    for router, score in (
        val_selectors.items()
    ):

        (
            budget,
            table,
        ) = select_budget(
            val,
            score,
            router,
        )

        selected_budgets[
            router
        ] = budget

        search_tables.append(
            table
        )

    (
        random_budget,
        random_search,
    ) = select_random_budget(
        val
    )

    selected_budgets[
        "Random"
    ] = random_budget

    search_tables.append(
        random_search
    )

    pd.concat(
        search_tables,
        ignore_index=True,
    ).to_csv(
        OUT
        / "validation_budget_search.csv",
        index=False,
    )

    # ==================================================
    # Refit learned baselines on train+val
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

    clf_ndcg_final = make_classifier(
        SEED + 100
    )

    clf_ndcg_final.fit(
        X_train_val,
        (
            train_val[
                "utility_ndcg_if_aug"
            ]
            > 1e-12
        ).astype(int),
    )

    test_ndcg_score = (
        positive_probability(
            clf_ndcg_final,
            X_test,
        )
    )

    clf_margin_final = make_classifier(
        SEED + 200
    )

    clf_margin_final.fit(
        X_train_val,
        train_val[
            "benefit_label"
        ].astype(int),
    )

    test_margin_score = (
        positive_probability(
            clf_margin_final,
            X_test,
        )
    )

    test_selectors = {
        "BenefitRF_NDCG":
            test_ndcg_score,

        "BenefitRF_Margin":
            test_margin_score,

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
    # Entire fixed-budget curves
    # ==================================================

    pieces = []

    for router, score in (
        test_selectors.items()
    ):

        for budget in BUDGETS:

            pieces.append(
                evaluate_with_scores(
                    test,
                    score,
                    budget,
                    router,
                )
            )

    for budget in BUDGETS:

        pieces.append(
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
            len(test),
            dtype=float,
        ),
        0.0,
        "NoAugment",
    )

    uniform = evaluate_with_scores(
        test,
        np.zeros(
            len(test),
            dtype=float,
        ),
        1.0,
        "UniformClean",
    )

    pieces.extend(
        [
            noaug,
            uniform,
        ]
    )

    case_df = pd.concat(
        pieces,
        ignore_index=True,
    )

    case_df.to_csv(
        OUT
        / "test_budget_case_metrics.csv",
        index=False,
    )

    curve = (
        case_df.groupby(
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

    curve.to_csv(
        OUT
        / "test_budget_curve.csv",
        index=False,
    )

    # ==================================================
    # Validation-selected baseline result
    # ==================================================

    selected = []

    for (
        router,
        budget
    ) in selected_budgets.items():

        x = case_df[
            (
                case_df[
                    "router"
                ] == router
            )
            &
            (
                np.isclose(
                    case_df[
                        "budget"
                    ],
                    budget,
                )
            )
        ].copy()

        selected.append(x)

    selected.extend(
        [
            noaug,
            uniform,
        ]
    )

    selected_case = pd.concat(
        selected,
        ignore_index=True,
    )

    selected_case.to_csv(
        OUT
        / "test_selected_case_metrics.csv",
        index=False,
    )

    summary = (
        selected_case.groupby(
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

    summary.to_csv(
        OUT
        / "test_selected_summary.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

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

        "selected_budgets":
            selected_budgets,
    }

    with (
        OUT
        / "baseline_manifest.json"
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
        "\n===== BASELINE SELECTED BUDGETS ====="
    )

    for k, v in (
        selected_budgets.items()
    ):
        print(
            f"{k:20s} {v:.2f}"
        )

    print(
        "\n===== BASELINE HELD-OUT TEST ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        OUT
    )


if __name__ == "__main__":
    main()
