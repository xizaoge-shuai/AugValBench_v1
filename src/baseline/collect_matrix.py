from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def discover_baseline_dirs():

    candidates = []

    # Existing 7B archived Amazon results.
    candidates.extend(
        ROOT.glob(
            "runs/*/results/"
            "baseline_*/"
        )
    )

    # General cross-model runs.
    candidates.extend(
        ROOT.glob(
            "baseline_runs/*/*/"
            "baseline_*/"
        )
    )

    unique = []

    seen = set()

    for p in candidates:

        manifest = (
            p
            / "baseline_manifest.json"
        )

        summary = (
            p
            / "test_selected_summary.csv"
        )

        if (
            manifest.exists()
            and summary.exists()
        ):

            key = str(
                p.resolve()
            )

            if key not in seen:
                seen.add(key)
                unique.append(p)

    return sorted(
        unique
    )


def main():

    rows = []

    dirs = (
        discover_baseline_dirs()
    )

    print(
        "discovered runs =",
        len(dirs)
    )

    for baseline_dir in dirs:

        import json

        with (
            baseline_dir
            / "baseline_manifest.json"
        ).open(
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

        summary = pd.read_csv(
            baseline_dir
            / "test_selected_summary.csv"
        )

        for _, r in (
            summary.iterrows()
        ):

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

                "source":
                    str(
                        baseline_dir
                        .relative_to(ROOT)
                    ),
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

    if len(out) == 0:
        print(
            "No completed runs."
        )
        return

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
        )[
            [
                "dataset",
                "model",
                "router",
                "budget",
                "ndcg10",
                "hr10",
                "mrr",
                "pairwise_win_rate",
                "n",
            ]
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
