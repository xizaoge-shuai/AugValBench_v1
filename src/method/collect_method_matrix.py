from pathlib import Path
import json

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def discover():

    paths = []

    paths.extend(
        ROOT.glob(
            "runs/*/results/"
            "method_risk_router/"
            "method_manifest.json"
        )
    )

    paths.extend(
        ROOT.glob(
            "baseline_runs/*/*/"
            "method_risk_router/"
            "method_manifest.json"
        )
    )

    return sorted(
        set(paths)
    )


def main():

    rows = []

    manifests = discover()

    print(
        "method runs =",
        len(manifests)
    )

    for mp in manifests:

        with mp.open(
            "r",
            encoding="utf-8",
        ) as f:

            manifest = json.load(f)

        dataset = manifest[
            "dataset"
        ]

        model = manifest[
            "model"
        ]

        method_dir = (
            mp.parent
        )

        result_root = (
            method_dir.parent
        )

        method = pd.read_csv(
            method_dir
            / "test_summary.csv"
        )

        for _, r in (
            method.iterrows()
        ):

            rows.append({
                "dataset":
                    dataset,

                "model":
                    model,

                "family":
                    "method",

                "router":
                    r[
                        "router"
                    ],

                "budget":
                    r[
                        "budget"
                    ],

                "ndcg10":
                    r[
                        "ndcg10"
                    ],

                "hr10":
                    r[
                        "hr10"
                    ],

                "mrr":
                    r[
                        "mrr"
                    ],

                "pairwise_win_rate":
                    r[
                        "pairwise_win_rate"
                    ],

                "augment_ratio":
                    r[
                        "augment_ratio"
                    ],

                "selected_harm_rate":
                    r[
                        "selected_harm_rate"
                    ],

                "selected_help_rate":
                    r[
                        "selected_help_rate"
                    ],
            })

        baseline_dir = (
            result_root
            / f"baseline_{dataset.lower()}"
        )

        baseline_path = (
            baseline_dir
            / "test_selected_summary.csv"
        )

        if baseline_path.exists():

            baseline = pd.read_csv(
                baseline_path
            )

            for _, r in (
                baseline.iterrows()
            ):

                rows.append({
                    "dataset":
                        dataset,

                    "model":
                        model,

                    "family":
                        "baseline",

                    "router":
                        r[
                            "router"
                        ],

                    "budget":
                        r[
                            "budget"
                        ],

                    "ndcg10":
                        r[
                            "ndcg10"
                        ],

                    "hr10":
                        r[
                            "hr10"
                        ],

                    "mrr":
                        r[
                            "mrr"
                        ],

                    "pairwise_win_rate":
                        r[
                            "pairwise_win_rate"
                        ],

                    "augment_ratio":
                        r[
                            "augment_ratio"
                        ],

                    "selected_harm_rate":
                        float("nan"),

                    "selected_help_rate":
                        float("nan"),
                })

    df = pd.DataFrame(
        rows
    )

    out_dir = (
        ROOT
        / "method_runs"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir
        / "risk_router_matrix.csv",
        index=False,
    )

    print()
    print(
        "===== FULL MATRIX ====="
    )

    print(
        df.sort_values(
            [
                "dataset",
                "model",
                "ndcg10",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        ).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    print()
    print(
        "===== COMPACT COMPARISON ====="
    )

    compare_rows = []

    for (
        dataset,
        model
    ), g in df.groupby(
        [
            "dataset",
            "model",
        ]
    ):

        usable_baseline = g[
            (
                g[
                    "family"
                ] == "baseline"
            )
            &
            (
                g[
                    "router"
                ] != "OracleNDCG"
            )
        ]

        risk = g[
            g[
                "router"
            ] == "RiskUtility"
        ]

        direct = g[
            g[
                "router"
            ] == "DirectExact"
        ]

        abstain = g[
            g[
                "router"
            ] == "DirectAbstain"
        ]

        if (
            len(risk) == 0
            or
            len(usable_baseline) == 0
        ):
            continue

        best = (
            usable_baseline
            .sort_values(
                "ndcg10",
                ascending=False,
            )
            .iloc[0]
        )

        rr = risk.iloc[0]

        de = (
            direct.iloc[0]
            if len(direct)
            else None
        )

        da = (
            abstain.iloc[0]
            if len(abstain)
            else None
        )

        compare_rows.append({
            "dataset":
                dataset,

            "model":
                model.split("/")[-1],

            "best_baseline":
                best["router"],

            "best_baseline_ndcg":
                best["ndcg10"],

            "direct_exact_ndcg":
                (
                    de["ndcg10"]
                    if de is not None
                    else float("nan")
                ),

            "direct_abstain_ndcg":
                (
                    da["ndcg10"]
                    if da is not None
                    else float("nan")
                ),

            "risk_utility_ndcg":
                rr["ndcg10"],

            "risk_actual_cost":
                rr[
                    "augment_ratio"
                ],

            "risk_harm_rate":
                rr[
                    "selected_harm_rate"
                ],

            "gain_vs_best_baseline":
                (
                    rr["ndcg10"]
                    -
                    best["ndcg10"]
                ),
        })

    compact = pd.DataFrame(
        compare_rows
    )

    compact.to_csv(
        out_dir
        / "risk_router_compact.csv",
        index=False,
    )

    print(
        compact.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )


if __name__ == "__main__":
    main()
