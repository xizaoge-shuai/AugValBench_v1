import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.utils import read_jsonl


EXPECTED = {
    "base",
    "clean",
    "conflict",
    "irrelevant",
    "partial",
    "redundant",
}


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source-result",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--perturb-result",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--benchmark-result",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--benchmark-data",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    args.benchmark_result.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = read_jsonl(
        args.benchmark_data
        / "eval_cases.jsonl"
    )

    case_ids = {
        int(x["case_id"])
        for x in cases
    }

    print(
        "benchmark cases =",
        len(case_ids)
    )

    # ==================================================
    # Existing Base/Clean
    # ==================================================

    source_scores = pd.read_csv(
        args.source_result
        / "rerank_scores.csv"
    )

    source_metrics = pd.read_csv(
        args.source_result
        / "case_metrics.csv"
    )

    source_scores = source_scores[
        source_scores[
            "case_id"
        ].isin(case_ids)
        &
        source_scores[
            "condition"
        ].isin(
            [
                "base",
                "clean",
            ]
        )
    ].copy()

    source_metrics = source_metrics[
        source_metrics[
            "case_id"
        ].isin(case_ids)
        &
        source_metrics[
            "condition"
        ].isin(
            [
                "base",
                "clean",
            ]
        )
    ].copy()

    # ==================================================
    # Newly computed perturbations
    # ==================================================

    perturb_scores = pd.read_csv(
        args.perturb_result
        / "rerank_scores.csv"
    )

    perturb_metrics = pd.read_csv(
        args.perturb_result
        / "case_metrics.csv"
    )

    perturb_scores = perturb_scores[
        perturb_scores[
            "case_id"
        ].isin(case_ids)
    ].copy()

    perturb_metrics = perturb_metrics[
        perturb_metrics[
            "case_id"
        ].isin(case_ids)
    ].copy()

    # ==================================================
    # Merge
    # ==================================================

    scores = pd.concat(
        [
            source_scores,
            perturb_scores,
        ],
        ignore_index=True,
    )

    metrics = pd.concat(
        [
            source_metrics,
            perturb_metrics,
        ],
        ignore_index=True,
    )

    score_keys = [
        x
        for x in [
            "case_id",
            "condition",
            "item_id",
        ]
        if x in scores.columns
    ]

    scores = scores.drop_duplicates(
        subset=score_keys,
        keep="last",
    )

    metrics = metrics.drop_duplicates(
        subset=[
            "case_id",
            "condition",
        ],
        keep="last",
    )

    got = set(
        metrics[
            "condition"
        ].unique()
    )

    print(
        "conditions =",
        sorted(got)
    )

    missing = (
        EXPECTED
        - got
    )

    if missing:
        raise RuntimeError(
            f"Missing conditions: {missing}"
        )

    counts = (
        metrics.groupby(
            "condition"
        )[
            "case_id"
        ]
        .nunique()
    )

    print(
        "\n===== CASE COUNT / CONDITION ====="
    )

    print(
        counts.to_string()
    )

    for condition in EXPECTED:

        if (
            int(counts[condition])
            != len(case_ids)
        ):
            raise RuntimeError(
                f"{condition}: "
                f"{counts[condition]} "
                f"!= {len(case_ids)}"
            )

    scores.to_csv(
        args.benchmark_result
        / "rerank_scores.csv",
        index=False,
    )

    metrics.to_csv(
        args.benchmark_result
        / "case_metrics.csv",
        index=False,
    )

    # analyze_gain uses knowledge information.
    src_knowledge = (
        args.source_result
        / "item_knowledge.csv"
    )

    if src_knowledge.exists():

        shutil.copy2(
            src_knowledge,
            args.benchmark_result
            / "item_knowledge.csv",
        )

    print(
        "\nSaved:",
        args.benchmark_result
    )


if __name__ == "__main__":
    main()
