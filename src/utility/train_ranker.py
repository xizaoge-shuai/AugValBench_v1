import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config


OUT = config.RESULT_DIR / "utility"


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )


    target = (
        df["utility_margin_if_aug"]
        .fillna(0)
        .astype(float)
    )


    drop_cols = [
        "benefit_label",
        "utility_ndcg_if_aug",
        "utility_pwr_if_aug",
        "utility_margin_if_aug",
        "case_id",
    ]


    X = df.drop(
        columns=[
            c for c in drop_cols
            if c in df.columns
        ]
    )


    X = X.select_dtypes(
        include=["number"]
    )


    groups = df["case_id"]


    splitter = GroupShuffleSplit(
        test_size=0.2,
        random_state=42,
    )


    train_idx, test_idx = next(
        splitter.split(
            X,
            target,
            groups,
        )
    )


    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]


    models = {

        "UtilityRF":
        Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                        min_samples_leaf=5,
                    )
                )
            ]
        ),


        "UtilityGBR":
        Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=42,
                        n_estimators=200,
                        learning_rate=0.03,
                        max_depth=3,
                    )
                )
            ]
        ),

    }


    metrics = []
    preds = []


    for name, model in models.items():

        print(
            "training",
            name
        )


        model.fit(
            X_train,
            y_train,
        )


        pred = model.predict(
            X_test
        )


        metrics.append(
            {
                "model": name,
                "mse":
                    mean_squared_error(
                        y_test,
                        pred,
                    ),
                "mae":
                    mean_absolute_error(
                        y_test,
                        pred,
                    ),
                "r2":
                    r2_score(
                        y_test,
                        pred,
                    )
            }
        )


        tmp = df.iloc[
            test_idx
        ].copy()

        tmp["pred_utility"] = pred

        tmp["model"] = name

        preds.append(tmp)


    pd.DataFrame(metrics).to_csv(
        OUT /
        "ranker_metrics.csv",
        index=False,
    )


    pd.concat(
        preds,
        ignore_index=True,
    ).to_csv(
        OUT /
        "ranker_predictions.csv",
        index=False,
    )


    print(
        pd.DataFrame(metrics)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
