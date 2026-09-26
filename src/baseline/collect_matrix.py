import json
from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


RUNS = [
    {
        "dataset":
            "All_Beauty",

        "model":
            "Qwen2.5-7B-Instruct",

        "result":
            ROOT
            / "runs"
            / "All_Beauty"
            / "results",
    },

    {
        "dataset":
            "Gift_Cards",

        "model":
            "Qwen2.5-7B-Instruct",

        "result":
            ROOT
            / "runs"
            / "Gift_Cards"
            / "results",
    },

    {
        "dataset":
            "All_Beauty",

        "model":
            "Qwen2.5-3B-Instruct",

        "result":
            ROOT
            / "baseline_runs"
            / "All_Beauty"
            / "Qwen2.5-3B-Instruct",
    },

    {
        "dataset":
            "Gift_Cards",

        "model":
            "Qwen2.5-3B-Instruct",

        "result":
            ROOT
            / "baseline_runs"
            / "Gift_Cards"
            / "Qwen2.5-3B-Instruct",
    },
]


def main():

    rows = []

    missing = []

    for run in RUNS:

        dataset = run[
            "dataset"
        ]

        model = run[
            "model"
        ]

        result = run[
            "result"
        ]

        baseline_dir = (
            result
            / f"baseline_{dataset.lower()}"
        )

        summary_path = (
            baseline_dir
            / "test_selected_summary.csv"
        )

        manifest_path = (
            baseline_dir
            / "baseline_manifest.json"
        )

        if not summary_path.exists():
            missing.append(
                str(summary_path)
            )
            continue

        summary = pd.read_csv(
            summary_path
        )

        for _, r in summary.iterrows():

            rows.append({
                "dataset":
                    dataset,

                "model":
                    model,

                "router":
                    r["router"],

                "budget":
                    r["budget"],

                "ndcg10":
                    r["ndcg10"],

                "hr10":
                    r["hr10"],

                "mrr":
                    r["mrr"],

                "pairwise_win_rate":
                    r[
                        "pairwise_win_rate"
                    ],

                "pairwise_margin":
                    r[
                        "pairwise_margin"
                    ],

                "n":
                    r["n"],
            })

    out = pd.DataFrame(
        rows
    )

    output_dir = (
        ROOT
        / "baseline_runs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_dir
        / "baseline_matrix.csv",
        index=False,
    )

    print(
        "\n===== BASELINE MATRIX ====="
    )

    if len(out):

        print(
            out.sort_values(
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
                index=False
            )
        )

    if missing:

        print(
            "\n===== NOT FINISHED ====="
        )

        for x in missing:
            print(x)


if __name__ == "__main__":
    main()
