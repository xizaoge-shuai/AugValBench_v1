import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


OUT_DIR = (
    config.RESULT_DIR
    / "benefit"
)


FEATURES = [
    "base_score",
    "base_rank_norm",
    "base_score_z",
    "base_top_gap",
    "case_base_score_mean",
    "case_base_score_std",

    "history_len",
    "history_pop_mean_log",
    "history_pop_std_log",

    "item_popularity_log",

    "knowledge_score",
    "knowledge_missing",
    "probe_accuracy",
    "n_probes",
    "mean_co_count",
    "mean_association_score",
    "mean_semantic_margin",
    "mean_abs_position_bias",

    "title_len",
    "description_len",
    "features_len",
    "meta_total_len",
    "category_present",
    "store_present",
]


def main():
    path = (
        OUT_DIR
        / "benefit_dataset.csv"
    )

    df = pd.read_csv(path)

    available = [
        x
        for x in FEATURES
        if x in df.columns
    ]

    print(
        "Features:",
        len(available),
    )

    print(
        "\n".join(
            available
        )
    )

    X = df[available]
    y = (
        df["benefit_label"]
        .astype(int)
    )

    groups = df[
        "case_id"
    ]

    n_cases = (
        groups.nunique()
    )

    n_splits = min(
        5,
        n_cases,
    )

    cv = GroupKFold(
        n_splits=n_splits
    )

    pred_lr = np.zeros(
        len(df),
        dtype=float,
    )

    pred_rf = np.zeros(
        len(df),
        dtype=float,
    )

    folds = np.zeros(
        len(df),
        dtype=int,
    )

    lr_importances = []
    rf_importances = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        )
    ):
        print(
            f"\n===== FOLD "
            f"{fold + 1}/{n_splits} ====="
        )

        X_train = X.iloc[
            train_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        print(
            "train cases =",
            df.iloc[
                train_idx
            ]["case_id"].nunique(),
        )

        print(
            "test cases =",
            df.iloc[
                test_idx
            ]["case_id"].nunique(),
        )

        print(
            "train positive rate =",
            y_train.mean(),
        )

        lr = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=2027,
                ),
            ),
        ])

        rf = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=2027 + fold,
                ),
            ),
        ])

        lr.fit(
            X_train,
            y_train,
        )

        rf.fit(
            X_train,
            y_train,
        )

        pred_lr[
            test_idx
        ] = lr.predict_proba(
            X_test
        )[:, 1]

        pred_rf[
            test_idx
        ] = rf.predict_proba(
            X_test
        )[:, 1]

        folds[
            test_idx
        ] = fold

        coef = (
            lr.named_steps[
                "model"
            ]
            .coef_[0]
        )

        lr_importances.append(
            np.abs(coef)
        )

        rf_importances.append(
            rf.named_steps[
                "model"
            ]
            .feature_importances_
        )

    df["fold"] = folds
    df["prob_lr"] = pred_lr
    df["prob_rf"] = pred_rf

    out_pred = (
        OUT_DIR
        / "predictions.csv"
    )

    df.to_csv(
        out_pred,
        index=False,
    )

    metrics = []

    for model_name, pred in [
        (
            "LogisticRegression",
            pred_lr,
        ),
        (
            "RandomForest",
            pred_rf,
        ),
    ]:
        auc = roc_auc_score(
            y,
            pred,
        )

        ap = average_precision_score(
            y,
            pred,
        )

        brier = brier_score_loss(
            y,
            pred,
        )

        metrics.append({
            "model": model_name,
            "auroc": auc,
            "average_precision": ap,
            "brier": brier,
            "positive_rate": y.mean(),
            "n": len(y),
            "cases": n_cases,
        })

    metric_df = pd.DataFrame(
        metrics
    )

    metric_df.to_csv(
        OUT_DIR
        / "predictor_metrics.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "feature": available,
        "lr_abs_coef": np.mean(
            lr_importances,
            axis=0,
        ),
        "rf_importance": np.mean(
            rf_importances,
            axis=0,
        ),
    })

    importance = importance.sort_values(
        "rf_importance",
        ascending=False,
    )

    importance.to_csv(
        OUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    print(
        "\n===== PREDICTOR METRICS ====="
    )

    print(
        metric_df.to_string(
            index=False
        )
    )

    print(
        "\n===== TOP FEATURES ====="
    )

    print(
        importance.head(
            15
        ).to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        out_pred,
    )


if __name__ == "__main__":
    main()
