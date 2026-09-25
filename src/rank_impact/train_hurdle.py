import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src import config
from src.utility.train_ranker import FEATURES


OUT = config.RESULT_DIR / "rank_impact"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

OBJECTIVES = {
    "pwr": "utility_pwr_if_aug",
    "ndcg": "utility_ndcg_if_aug",
}

FORBIDDEN = {
    "is_target",
    "clean_score",
    "benefit_label",
    "utility_margin_if_aug",
    "utility_pwr_if_aug",
    "utility_ndcg_if_aug",
}


def safe_spearman(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if (
        len(y) < 3
        or np.std(y) <= 1e-15
        or np.std(pred) <= 1e-15
    ):
        return np.nan, np.nan

    return spearmanr(y, pred)


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


def class_probabilities(model, X):
    """
    返回：
      p_harm
      p_neutral
      p_benefit

    class mapping:
      0 = harmful
      1 = neutral
      2 = beneficial
    """
    probs = model.predict_proba(X)

    classes = (
        model.named_steps["model"]
        .classes_
    )

    output = {
        0: np.zeros(len(X)),
        1: np.zeros(len(X)),
        2: np.zeros(len(X)),
    }

    for j, cls in enumerate(classes):
        output[int(cls)] = probs[:, j]

    return (
        output[0],
        output[1],
        output[2],
    )


def main():
    path = (
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )

    df = pd.read_csv(path)

    available = [
        f for f in FEATURES
        if f in df.columns
    ]

    leaking = (
        set(available)
        & FORBIDDEN
    )

    if leaking:
        raise RuntimeError(
            f"LEAKAGE: {leaking}"
        )

    print("===== SAFE FEATURES =====")
    print("count =", len(available))

    for f in available:
        print(f)

    print("\n===== FORBIDDEN INTERSECTION =====")
    print(leaking)

    X = df[available].copy()
    groups = df["case_id"]

    cv = GroupKFold(
        n_splits=5
    )

    output = df.copy()

    metrics = []

    for objective, target_col in OBJECTIVES.items():

        print(
            "\n"
            + "=" * 80
        )
        print(
            "OBJECTIVE:",
            objective
        )
        print(
            "=" * 80
        )

        y = (
            df[target_col]
            .fillna(0.0)
            .astype(float)
        )

        y_np = y.to_numpy()

        label = np.ones(
            len(df),
            dtype=int,
        )

        label[
            y_np < -EPS
        ] = 0

        label[
            y_np > EPS
        ] = 2

        print(
            "harmful =",
            int(
                np.sum(label == 0)
            )
        )

        print(
            "neutral =",
            int(
                np.sum(label == 1)
            )
        )

        print(
            "beneficial =",
            int(
                np.sum(label == 2)
            )
        )

        pred_hurdle = np.zeros(
            len(df),
            dtype=float,
        )

        pred_direct = np.zeros(
            len(df),
            dtype=float,
        )

        p_harm_all = np.zeros(
            len(df),
            dtype=float,
        )

        p_neutral_all = np.zeros(
            len(df),
            dtype=float,
        )

        p_benefit_all = np.zeros(
            len(df),
            dtype=float,
        )

        pos_mag_all = np.zeros(
            len(df),
            dtype=float,
        )

        neg_mag_all = np.zeros(
            len(df),
            dtype=float,
        )

        fold_ids = np.full(
            len(df),
            -1,
            dtype=int,
        )

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            cv.split(
                X,
                label,
                groups,
            )
        ):

            print(
                f"\n--- "
                f"{objective} "
                f"fold {fold + 1}/5 ---"
            )

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ].to_numpy()

            label_train = label[
                train_idx
            ]

            classifier = make_classifier(
                2027
                + fold
                + (
                    100
                    if objective == "ndcg"
                    else 0
                )
            )

            classifier.fit(
                X_train,
                label_train,
            )

            (
                p_harm,
                p_neutral,
                p_benefit,
            ) = class_probabilities(
                classifier,
                X_test,
            )

            # ----------------------------------
            # positive magnitude
            # ----------------------------------

            pos_mask = (
                y_train > EPS
            )

            if np.sum(pos_mask) >= 10:
                positive_model = make_regressor(
                    3027 + fold
                )

                positive_model.fit(
                    X_train.iloc[
                        np.where(
                            pos_mask
                        )[0]
                    ],
                    y_train[
                        pos_mask
                    ],
                )

                pos_mag = positive_model.predict(
                    X_test
                )

                pos_mag = np.maximum(
                    pos_mag,
                    0.0,
                )

            else:
                pos_mag = np.full(
                    len(test_idx),
                    (
                        np.mean(
                            y_train[
                                pos_mask
                            ]
                        )
                        if np.any(
                            pos_mask
                        )
                        else 0.0
                    ),
                )

            # ----------------------------------
            # harmful magnitude
            # ----------------------------------

            neg_mask = (
                y_train < -EPS
            )

            if np.sum(neg_mask) >= 10:
                negative_model = make_regressor(
                    4027 + fold
                )

                negative_model.fit(
                    X_train.iloc[
                        np.where(
                            neg_mask
                        )[0]
                    ],
                    -y_train[
                        neg_mask
                    ],
                )

                neg_mag = negative_model.predict(
                    X_test
                )

                neg_mag = np.maximum(
                    neg_mag,
                    0.0,
                )

            else:
                neg_mag = np.full(
                    len(test_idx),
                    (
                        np.mean(
                            -y_train[
                                neg_mask
                            ]
                        )
                        if np.any(
                            neg_mask
                        )
                        else 0.0
                    ),
                )

            expected_utility = (
                p_benefit
                * pos_mag
                -
                p_harm
                * neg_mag
            )

            # ----------------------------------
            # direct regression baseline
            # ----------------------------------

            direct_model = make_regressor(
                5027
                + fold
                + (
                    100
                    if objective == "ndcg"
                    else 0
                )
            )

            direct_model.fit(
                X_train,
                y_train,
            )

            direct_pred = (
                direct_model.predict(
                    X_test
                )
            )

            pred_hurdle[
                test_idx
            ] = expected_utility

            pred_direct[
                test_idx
            ] = direct_pred

            p_harm_all[
                test_idx
            ] = p_harm

            p_neutral_all[
                test_idx
            ] = p_neutral

            p_benefit_all[
                test_idx
            ] = p_benefit

            pos_mag_all[
                test_idx
            ] = pos_mag

            neg_mag_all[
                test_idx
            ] = neg_mag

            fold_ids[
                test_idx
            ] = fold

        output[
            f"pred_hurdle_{objective}"
        ] = pred_hurdle

        output[
            f"pred_direct_{objective}"
        ] = pred_direct

        output[
            f"p_harm_{objective}"
        ] = p_harm_all

        output[
            f"p_neutral_{objective}"
        ] = p_neutral_all

        output[
            f"p_benefit_{objective}"
        ] = p_benefit_all

        output[
            f"pred_positive_mag_{objective}"
        ] = pos_mag_all

        output[
            f"pred_negative_mag_{objective}"
        ] = neg_mag_all

        output[
            "fold"
        ] = fold_ids

        for model_name, pred in [
            (
                "Hurdle",
                pred_hurdle,
            ),
            (
                "DirectRF",
                pred_direct,
            ),
        ]:

            rho_all, p_all = (
                safe_spearman(
                    y_np,
                    pred,
                )
            )

            nonzero = (
                np.abs(
                    y_np
                )
                > EPS
            )

            rho_nz, p_nz = (
                safe_spearman(
                    y_np[
                        nonzero
                    ],
                    pred[
                        nonzero
                    ],
                )
            )

            benefit_truth = (
                y_np > EPS
            ).astype(int)

            harm_truth = (
                y_np < -EPS
            ).astype(int)

            benefit_ap = (
                average_precision_score(
                    benefit_truth,
                    pred,
                )
                if benefit_truth.sum() > 0
                else np.nan
            )

            harm_ap = (
                average_precision_score(
                    harm_truth,
                    -pred,
                )
                if harm_truth.sum() > 0
                else np.nan
            )

            metrics.append({
                "objective":
                    objective,
                "model":
                    model_name,
                "spearman_all":
                    rho_all,
                "spearman_all_p":
                    p_all,
                "spearman_nonzero":
                    rho_nz,
                "spearman_nonzero_p":
                    p_nz,
                "benefit_ap":
                    benefit_ap,
                "harm_ap":
                    harm_ap,
                "benefit_rate":
                    benefit_truth.mean(),
                "harm_rate":
                    harm_truth.mean(),
                "neutral_rate":
                    1.0
                    - benefit_truth.mean()
                    - harm_truth.mean(),
            })

    output.to_csv(
        OUT
        / "rank_impact_predictions.csv",
        index=False,
    )

    metric_df = pd.DataFrame(
        metrics
    )

    metric_df.to_csv(
        OUT
        / "rank_impact_metrics.csv",
        index=False,
    )

    print(
        "\n===== OOF RANK-IMPACT METRICS ====="
    )

    print(
        metric_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        OUT
        / "rank_impact_predictions.csv"
    )


if __name__ == "__main__":
    main()
