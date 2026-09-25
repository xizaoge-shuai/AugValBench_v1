import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src import config


OUT = config.RESULT_DIR / "utility"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 只能使用 augmentation 发生之前能够获得的特征
# ============================================================

FEATURES = [
    # recommendation state
    "base_score",
    "base_rank_norm",
    "base_score_z",
    "base_top_gap",
    "case_base_score_mean",
    "case_base_score_std",

    # user state
    "history_len",
    "history_pop_mean_log",
    "history_pop_std_log",

    # item state
    "item_popularity_log",

    # knowledge state
    "knowledge_score",
    "knowledge_missing",
    "probe_accuracy",
    "n_probes",
    "mean_co_count",
    "mean_association_score",
    "mean_semantic_margin",
    "mean_abs_position_bias",

    # metadata availability / quality
    "title_len",
    "description_len",
    "features_len",
    "meta_total_len",
    "category_present",
    "store_present",
]


FORBIDDEN = {
    "is_target",
    "clean_score",
    "benefit_label",
    "utility_ndcg_if_aug",
    "utility_pwr_if_aug",
    "utility_margin_if_aug",
}


def regression_metrics(y, pred, subset):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)

    if len(y) >= 2:
        r2 = r2_score(y, pred)
    else:
        r2 = np.nan

    if len(y) >= 3 and np.std(y) > 0 and np.std(pred) > 0:
        rho, p = spearmanr(y, pred)
    else:
        rho, p = np.nan, np.nan

    return {
        "subset": subset,
        "n": len(y),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "spearman_rho": rho,
        "spearman_p": p,
    }


def main():

    path = (
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )

    df = pd.read_csv(path)

    available = [
        x for x in FEATURES
        if x in df.columns
    ]

    leaking = set(available) & FORBIDDEN

    if leaking:
        raise RuntimeError(
            f"Feature leakage detected: {leaking}"
        )

    print("===== SAFE FEATURES =====")
    print("feature count =", len(available))

    for x in available:
        print(x)

    print("\n===== MISSING VALUES =====")

    missing = (
        df[available]
        .isna()
        .sum()
    )

    print(
        missing[
            missing > 0
        ].to_string()
    )

    X = df[available].copy()

    y = (
        df["utility_margin_if_aug"]
        .astype(float)
    )

    groups = df["case_id"]

    cv = GroupKFold(
        n_splits=5
    )

    pred_rf = np.zeros(
        len(df),
        dtype=float,
    )

    pred_gbr = np.zeros(
        len(df),
        dtype=float,
    )

    fold_ids = np.full(
        len(df),
        -1,
        dtype=int,
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups)
    ):

        print(
            f"\n===== FOLD {fold + 1}/5 ====="
        )

        print(
            "train cases =",
            df.iloc[train_idx]["case_id"].nunique()
        )

        print(
            "test cases  =",
            df.iloc[test_idx]["case_id"].nunique()
        )

        rf = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=4,
                    random_state=2027 + fold,
                    n_jobs=-1,
                ),
            ),
        ])

        # Huber loss 对我们现在这种 heavy-tail utility
        # 比普通 squared error 更稳。
        gbr = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                GradientBoostingRegressor(
                    loss="huber",
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=4,
                    random_state=2027 + fold,
                ),
            ),
        ])

        rf.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        gbr.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        pred_rf[test_idx] = rf.predict(
            X.iloc[test_idx]
        )

        pred_gbr[test_idx] = gbr.predict(
            X.iloc[test_idx]
        )

        fold_ids[test_idx] = fold

    out = df.copy()

    out["fold"] = fold_ids
    out["pred_utility_rf"] = pred_rf
    out["pred_utility_gbr"] = pred_gbr

    out.to_csv(
        OUT / "ranker_predictions.csv",
        index=False,
    )

    metric_rows = []

    for model_name, pred in [
        ("UtilityRF", pred_rf),
        ("UtilityGBR", pred_gbr),
    ]:

        row = {
            "model": model_name,
            **regression_metrics(
                y,
                pred,
                "all",
            ),
        }

        metric_rows.append(row)

        target_mask = (
            df["is_target"] == 1
        ).to_numpy()

        negative_mask = ~target_mask

        metric_rows.append({
            "model": model_name,
            **regression_metrics(
                y[target_mask],
                pred[target_mask],
                "target",
            ),
        })

        metric_rows.append({
            "model": model_name,
            **regression_metrics(
                y[negative_mask],
                pred[negative_mask],
                "negative",
            ),
        })

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        OUT / "ranker_metrics.csv",
        index=False,
    )

    print(
        "\n===== OOF REGRESSION METRICS ====="
    )

    print(
        metrics.to_string(index=False)
    )

    print(
        "\nSaved:",
        OUT / "ranker_predictions.csv"
    )


if __name__ == "__main__":
    main()
