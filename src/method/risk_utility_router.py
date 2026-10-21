import json
from pathlib import Path

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
    / "method_risk_router"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 2027

EPS = 1e-8

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

LAMBDAS = [
    0.0,
    0.25,
    0.50,
    1.0,
    2.0,
    4.0,
]


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


def evaluate_policy(
    frame,
    selector_score,
    budget,
    router,
    abstain,
):

    work = frame.copy()

    work[
        "_selector_score"
    ] = np.asarray(
        selector_score,
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

        selector = (
            case[
                "_selector_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if budget <= 0:

            selected = np.asarray(
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

            order = np.argsort(
                -selector,
                kind="stable",
            )

            if abstain:

                order = order[
                    selector[
                        order
                    ] > 0.0
                ]

            selected = order[
                :k
            ]

        base_scores = (
            case[
                "base_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        clean_scores = (
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

        final_scores = (
            base_scores.copy()
        )

        final_scores[
            selected
        ] = clean_scores[
            selected
        ]

        metrics = (
            expected_tie_metrics(
                final_scores,
                labels,
            )
        )

        true_utility = (
            case[
                "utility_ndcg_if_aug"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(selected) > 0:

            selected_utility = (
                true_utility[
                    selected
                ]
            )

            selected_mean_utility = (
                float(
                    selected_utility.mean()
                )
            )

            selected_harm_rate = (
                float(
                    (
                        selected_utility
                        < -EPS
                    ).mean()
                )
            )

            selected_help_rate = (
                float(
                    (
                        selected_utility
                        > EPS
                    ).mean()
                )
            )

        else:

            selected_mean_utility = 0.0
            selected_harm_rate = 0.0
            selected_help_rate = 0.0

        rows.append({
            "case_id":
                int(case_id),

            "router":
                router,

            "budget":
                float(budget),

            "selected_count":
                int(
                    len(selected)
                ),

            "augment_ratio":
                float(
                    len(selected)
                    / n
                ),

            "selected_mean_individual_utility":
                selected_mean_utility,

            "selected_harm_rate":
                selected_harm_rate,

            "selected_help_rate":
                selected_help_rate,

            **metrics,
        })

    return pd.DataFrame(
        rows
    )


def summarize(
    case_df,
):

    return {
        "ndcg10":
            float(
                case_df[
                    "ndcg10"
                ].mean()
            ),

        "hr10":
            float(
                case_df[
                    "hr10"
                ].mean()
            ),

        "mrr":
            float(
                case_df[
                    "mrr"
                ].mean()
            ),

        "pairwise_win_rate":
            float(
                case_df[
                    "pairwise_win_rate"
                ].mean()
            ),

        "pairwise_margin":
            float(
                case_df[
                    "pairwise_margin"
                ].mean()
            ),

        "augment_ratio":
            float(
                case_df[
                    "augment_ratio"
                ].mean()
            ),

        "selected_harm_rate":
            float(
                case_df[
                    "selected_harm_rate"
                ].mean()
            ),

        "selected_help_rate":
            float(
                case_df[
                    "selected_help_rate"
                ].mean()
            ),
    }


def choose_best(
    table,
):

    # Main objective: validation NDCG.
    # Ties -> lower augmentation cost,
    # then smaller risk coefficient.
    return (
        table.sort_values(
            [
                "val_ndcg10",
                "val_augment_ratio",
                "lambda",
                "budget",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .iloc[0]
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

    if not data_path.exists():

        raise FileNotFoundError(
            data_path
        )

    if not split_path.exists():

        raise FileNotFoundError(
            split_path
        )

    df = pd.read_csv(
        data_path
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

    test = (
        df[
            df[
                "split"
            ] == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "===== DATA ====="
    )

    print(
        "dataset =",
        config.DATASET_NAME
    )

    print(
        "model =",
        config.MODEL_NAME
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

    X_train = (
        train[
            features
        ]
    )

    X_val = (
        val[
            features
        ]
    )

    # ==================================================
    # Utility head
    # ==================================================

    utility_model = (
        make_regressor(
            SEED
        )
    )

    utility_model.fit(
        X_train,
        train[
            "utility_ndcg_if_aug"
        ],
    )

    mu_val = (
        utility_model.predict(
            X_val
        )
    )

    # ==================================================
    # Harm-risk head
    # ==================================================

    y_harm_train = (
        train[
            "utility_ndcg_if_aug"
        ]
        < -EPS
    ).astype(int)

    harm_model = (
        make_classifier(
            SEED + 100
        )
    )

    harm_model.fit(
        X_train,
        y_harm_train,
    )

    harm_prob_val = (
        positive_probability(
            harm_model,
            X_val,
        )
    )

    negative_train = (
        train.loc[
            train[
                "utility_ndcg_if_aug"
            ] < -EPS,
            "utility_ndcg_if_aug",
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        negative_train
    ) == 0:

        harm_scale = 0.0

    else:

        harm_scale = float(
            np.mean(
                negative_train
            )
        )

    print(
        "train harm prevalence =",
        float(
            y_harm_train.mean()
        )
    )

    print(
        "harm magnitude scale =",
        harm_scale
    )

    # ==================================================
    # Validation search
    # ==================================================

    validation_rows = []

    # ------------------------------
    # DirectNDCG exact-budget
    # ------------------------------

    for budget in BUDGETS:

        cases = (
            evaluate_policy(
                val,
                mu_val,
                budget,
                "DirectExact",
                abstain=False,
            )
        )

        s = summarize(
            cases
        )

        validation_rows.append({
            "router":
                "DirectExact",

            "lambda":
                0.0,

            "budget":
                budget,

            **{
                f"val_{k}":
                    v
                for k, v in s.items()
            },
        })

    # ------------------------------
    # DirectNDCG + abstention
    # ------------------------------

    for budget in BUDGETS:

        cases = (
            evaluate_policy(
                val,
                mu_val,
                budget,
                "DirectAbstain",
                abstain=True,
            )
        )

        s = summarize(
            cases
        )

        validation_rows.append({
            "router":
                "DirectAbstain",

            "lambda":
                0.0,

            "budget":
                budget,

            **{
                f"val_{k}":
                    v
                for k, v in s.items()
            },
        })

    # ------------------------------
    # Risk-calibrated utility
    # ------------------------------

    for lam in LAMBDAS:

        risk_val = (
            mu_val
            -
            lam
            * harm_prob_val
            * harm_scale
        )

        for budget in BUDGETS:

            cases = (
                evaluate_policy(
                    val,
                    risk_val,
                    budget,
                    "RiskUtility",
                    abstain=True,
                )
            )

            s = summarize(
                cases
            )

            validation_rows.append({
                "router":
                    "RiskUtility",

                "lambda":
                    lam,

                "budget":
                    budget,

                **{
                    f"val_{k}":
                        v
                    for k, v in s.items()
                },
            })

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        OUT
        / "validation_search.csv",
        index=False,
    )

    selected = {}

    for router in [
        "DirectExact",
        "DirectAbstain",
        "RiskUtility",
    ]:

        table = (
            validation[
                validation[
                    "router"
                ] == router
            ]
        )

        best = choose_best(
            table
        )

        selected[
            router
        ] = {
            "budget":
                float(
                    best[
                        "budget"
                    ]
                ),

            "lambda":
                float(
                    best[
                        "lambda"
                    ]
                ),

            "val_ndcg10":
                float(
                    best[
                        "val_ndcg10"
                    ]
                ),

            "val_augment_ratio":
                float(
                    best[
                        "val_augment_ratio"
                    ]
                ),
        }

    print()
    print(
        "===== VALIDATION SELECTION ====="
    )

    for router, x in (
        selected.items()
    ):

        print(
            f"{router:15s} "
            f"budget={x['budget']:.2f} "
            f"lambda={x['lambda']:.2f} "
            f"NDCG={x['val_ndcg10']:.6f} "
            f"actual_cost="
            f"{x['val_augment_ratio']:.4f}"
        )

    # ==================================================
    # Refit train + val
    # ==================================================

    train_val = pd.concat(
        [
            train,
            val,
        ],
        ignore_index=True,
    )

    X_train_val = (
        train_val[
            features
        ]
    )

    X_test = (
        test[
            features
        ]
    )

    utility_final = (
        make_regressor(
            SEED
        )
    )

    utility_final.fit(
        X_train_val,
        train_val[
            "utility_ndcg_if_aug"
        ],
    )

    mu_test = (
        utility_final.predict(
            X_test
        )
    )

    y_harm_train_val = (
        train_val[
            "utility_ndcg_if_aug"
        ]
        < -EPS
    ).astype(int)

    harm_final = (
        make_classifier(
            SEED + 100
        )
    )

    harm_final.fit(
        X_train_val,
        y_harm_train_val,
    )

    harm_prob_test = (
        positive_probability(
            harm_final,
            X_test,
        )
    )

    negative_train_val = (
        train_val.loc[
            train_val[
                "utility_ndcg_if_aug"
            ] < -EPS,
            "utility_ndcg_if_aug",
        ]
        .abs()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        negative_train_val
    ) == 0:

        harm_scale_final = 0.0

    else:

        harm_scale_final = (
            float(
                np.mean(
                    negative_train_val
                )
            )
        )

    # ==================================================
    # Held-out test
    # ==================================================

    parts = []

    direct_exact_cfg = (
        selected[
            "DirectExact"
        ]
    )

    parts.append(
        evaluate_policy(
            test,
            mu_test,
            direct_exact_cfg[
                "budget"
            ],
            "DirectExact",
            abstain=False,
        )
    )

    direct_abstain_cfg = (
        selected[
            "DirectAbstain"
        ]
    )

    parts.append(
        evaluate_policy(
            test,
            mu_test,
            direct_abstain_cfg[
                "budget"
            ],
            "DirectAbstain",
            abstain=True,
        )
    )

    risk_cfg = (
        selected[
            "RiskUtility"
        ]
    )

    risk_score_test = (
        mu_test
        -
        risk_cfg[
            "lambda"
        ]
        * harm_prob_test
        * harm_scale_final
    )

    parts.append(
        evaluate_policy(
            test,
            risk_score_test,
            risk_cfg[
                "budget"
            ],
            "RiskUtility",
            abstain=True,
        )
    )

    parts.append(
        evaluate_policy(
            test,
            np.zeros(
                len(test),
                dtype=float,
            ),
            0.0,
            "NoAugment",
            abstain=False,
        )
    )

    parts.append(
        evaluate_policy(
            test,
            np.zeros(
                len(test),
                dtype=float,
            ),
            1.0,
            "UniformClean",
            abstain=False,
        )
    )

    case_metrics = pd.concat(
        parts,
        ignore_index=True,
    )

    case_metrics.to_csv(
        OUT
        / "test_case_metrics.csv",
        index=False,
    )

    summary_rows = []

    for router, x in (
        case_metrics.groupby(
            "router",
            sort=False,
        )
    ):

        s = summarize(
            x
        )

        summary_rows.append({
            "dataset":
                config.DATASET_NAME,

            "model":
                config.MODEL_NAME,

            "router":
                router,

            "budget":
                float(
                    x[
                        "budget"
                    ].iloc[0]
                ),

            "n":
                int(
                    x[
                        "case_id"
                    ].nunique()
                ),

            **s,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        "ndcg10",
        ascending=False,
    )

    summary.to_csv(
        OUT
        / "test_summary.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

        "features":
            features,

        "eps":
            EPS,

        "harm_scale_train":
            harm_scale,

        "harm_scale_train_val":
            harm_scale_final,

        "selected":
            selected,

        "utility_model": {
            "type":
                "RandomForestRegressor",

            "n_estimators":
                600,

            "max_depth":
                12,

            "min_samples_leaf":
                3,
        },

        "harm_model": {
            "type":
                "RandomForestClassifier",

            "n_estimators":
                600,

            "max_depth":
                12,

            "min_samples_leaf":
                3,
        },
    }

    with (
        OUT
        / "method_manifest.json"
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
        "===== HELD-OUT TEST ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUT
    )


if __name__ == "__main__":
    main()
