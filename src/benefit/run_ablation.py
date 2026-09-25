import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src import config


OUT_DIR = (
    config.RESULT_DIR
    / "benefit"
)


GROUPS = {
    "State": [
        "base_score",
        "base_rank_norm",
        "base_score_z",
        "base_top_gap",
        "case_base_score_mean",
        "case_base_score_std",
    ],

    "User": [
        "history_len",
        "history_pop_mean_log",
        "history_pop_std_log",
    ],

    "Item": [
        "item_popularity_log",
        "title_len",
        "description_len",
        "features_len",
        "meta_total_len",
        "category_present",
        "store_present",
    ],

    "Knowledge": [
        "knowledge_score",
        "knowledge_missing",
        "probe_accuracy",
        "n_probes",
        "mean_co_count",
        "mean_association_score",
        "mean_semantic_margin",
        "mean_abs_position_bias",
    ],
}


CONFIGS = {
    "KnowledgeOnly": [
        "Knowledge",
    ],

    "StateOnly": [
        "State",
    ],

    "ItemUserOnly": [
        "Item",
        "User",
    ],

    "FullWithoutKnowledge": [
        "State",
        "Item",
        "User",
    ],

    "Full": [
        "State",
        "Item",
        "User",
        "Knowledge",
    ],
}


def flatten(group_names):
    out = []

    for group in group_names:
        out.extend(
            GROUPS[group]
        )

    return list(
        dict.fromkeys(
            out
        )
    )


def main():
    df = pd.read_csv(
        OUT_DIR
        / "benefit_dataset.csv"
    )

    y = (
        df["benefit_label"]
        .astype(int)
    )

    groups = df[
        "case_id"
    ]

    cv = GroupKFold(
        n_splits=5
    )

    rows = []

    prevalence = float(
        y.mean()
    )

    rows.append({
        "setting": "ConstantPrevalence",
        "n_features": 0,
        "auroc": 0.5,
        "average_precision": prevalence,
        "brier": prevalence * (
            1.0 - prevalence
        ),
    })

    for setting, group_names in (
        CONFIGS.items()
    ):
        features = [
            f
            for f
            in flatten(
                group_names
            )
            if f in df.columns
        ]

        X = df[
            features
        ]

        pred = np.zeros(
            len(df),
            dtype=float,
        )

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
            model = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=2027 + fold,
                    ),
                ),
            ])

            model.fit(
                X.iloc[
                    train_idx
                ],
                y.iloc[
                    train_idx
                ],
            )

            pred[
                test_idx
            ] = model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]

        rows.append({
            "setting": setting,
            "n_features": len(
                features
            ),
            "auroc": roc_auc_score(
                y,
                pred,
            ),
            "average_precision":
                average_precision_score(
                    y,
                    pred,
                ),
            "brier":
                brier_score_loss(
                    y,
                    pred,
                ),
        })

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        "auroc",
        ascending=False,
    )

    out.to_csv(
        OUT_DIR
        / "ablation_metrics.csv",
        index=False,
    )

    print(
        "\n===== FEATURE ABLATION ====="
    )

    print(
        out.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
