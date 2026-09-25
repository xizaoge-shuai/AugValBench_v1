import pandas as pd
import numpy as np

from src import config


OUT = config.RESULT_DIR / "utility"


BUDGETS=[
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]


def topk(x,k):

    return np.argsort(
        -x,
        kind="stable"
    )[:k]


def main():

    df=pd.read_csv(
        OUT /
        "ranker_predictions.csv"
    )


    rows=[]


    for model,g in df.groupby(
        "model"
    ):

        for cid,case in g.groupby(
            "case_id"
        ):

            case=case.reset_index(
                drop=True
            )


            utility=case[
                "utility_margin_if_aug"
            ].values


            score=case[
                "pred_utility"
            ].values


            for b in BUDGETS:

                k=max(
                    1,
                    int(
                        len(case)*b
                    )
                )


                idx=topk(
                    score,
                    k
                )


                selected=utility[idx]


                rows.append(
                    {
                        "model":model,
                        "case_id":cid,
                        "budget":b,
                        "utility":
                            selected.sum(),
                        "positive_capture":
                            np.maximum(
                                selected,
                                0
                            ).sum()
                    }
                )


    result=pd.DataFrame(rows)


    result.to_csv(
        OUT /
        "ranker_budget.csv",
        index=False,
    )


    print(
        result.groupby(
            [
                "model",
                "budget"
            ]
        )
        [
            [
                "utility",
                "positive_capture"
            ]
        ]
        .mean()
        .to_string()
    )


if __name__=="__main__":
    main()
