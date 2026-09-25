import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config


OUT_DIR = config.RESULT_DIR / "benefit"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RATIOS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

RANDOM_REPEATS = 30


def expected_tie_metrics(scores, labels):
    """
    对 score tie 使用随机 tie-breaking 下的期望指标。
    这样不会因为 pandas stable/unstable sort 导致结果变化。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    target_idx = np.where(labels == 1)[0]

    if len(target_idx) != 1:
        raise ValueError("Each case must have exactly one target.")

    target_idx = int(target_idx[0])
    target_score = scores[target_idx]

    neg_scores = scores[labels == 0]

    higher = int(np.sum(scores > target_score))

    tied_total = int(np.sum(scores == target_score))

    first_rank = higher + 1
    last_rank = higher + tied_total

    positions = np.arange(
        first_rank,
        last_rank + 1,
        dtype=float,
    )

    expected_rank = float(
        positions.mean()
    )

    discounts = np.where(
        positions <= 10,
        1.0 / np.log2(positions + 1.0),
        0.0,
    )

    ndcg10 = float(
        discounts.mean()
    )

    hr10 = float(
        np.mean(positions <= 10)
    )

    mrr = float(
        np.mean(1.0 / positions)
    )

    wins = (
        target_score > neg_scores
    ).astype(float)

    ties = (
        target_score == neg_scores
    ).astype(float)

    pwr = float(
        np.mean(
            wins + 0.5 * ties
        )
    )

    margin = float(
        target_score
        - np.mean(neg_scores)
    )

    return {
        "ndcg10": ndcg10,
        "hr10": hr10,
        "mrr": mrr,
        "rank": expected_rank,
        "pairwise_win_rate": pwr,
        "pairwise_margin": margin,
    }


def select_top(values, k, largest=True):
    values = np.asarray(values, dtype=float)

    if largest:
        order = np.argsort(
            -values,
            kind="stable",
        )
    else:
        order = np.argsort(
            values,
            kind="stable",
        )

    mask = np.zeros(
        len(values),
        dtype=bool,
    )

    if k > 0:
        mask[order[:k]] = True

    return mask


def evaluate_mask(g, mask):
    score = np.where(
        mask,
        g["clean_score"].to_numpy(dtype=float),
        g["base_score"].to_numpy(dtype=float),
    )

    labels = g[
        "is_target"
    ].to_numpy(dtype=int)

    return expected_tie_metrics(
        score,
        labels,
    )


def main():
    df = pd.read_csv(
        OUT_DIR / "predictions.csv"
    )

    rows = []

    # NoAugment，budget=0
    for case_id, g in df.groupby(
        "case_id",
        sort=False,
    ):
        g = g.reset_index(
            drop=True
        )

        m = evaluate_mask(
            g,
            np.zeros(
                len(g),
                dtype=bool,
            ),
        )

        rows.append({
            "case_id": case_id,
            "router": "NoAugment",
            "budget": 0.0,
            **m,
        })

    for ratio in RATIOS:
        print(
            f"budget = {ratio:.2f}"
        )

        for case_id, g in df.groupby(
            "case_id",
            sort=False,
        ):
            g = g.reset_index(
                drop=True
            )

            n = len(g)

            k = max(
                1,
                int(
                    round(
                        ratio * n
                    )
                ),
            )

            selectors = {}

            selectors["BenefitLR"] = select_top(
                g["prob_lr"],
                k,
                largest=True,
            )

            selectors["BenefitRF"] = select_top(
                g["prob_rf"],
                k,
                largest=True,
            )

            selectors["Tail"] = select_top(
                g["item_popularity_log"],
                k,
                largest=False,
            )

            knowledge = (
                g["knowledge_score"]
                .fillna(1.0)
            )

            selectors["LowKnowledge"] = select_top(
                knowledge,
                k,
                largest=False,
            )

            selectors["LabelOracle"] = select_top(
                g["utility_margin_if_aug"],
                k,
                largest=True,
            )

            for name, mask in selectors.items():
                m = evaluate_mask(
                    g,
                    mask,
                )

                rows.append({
                    "case_id": case_id,
                    "router": name,
                    "budget": ratio,
                    **m,
                })

            # Random 做 30 次，先对同一个 case 取平均，
            # 避免一次随机采样带来太大噪声。
            random_metrics = []

            for rep in range(
                RANDOM_REPEATS
            ):
                rng = random.Random(
                    2027
                    + case_id * 1009
                    + rep * 97
                    + int(ratio * 1000)
                )

                ids = list(
                    range(n)
                )

                rng.shuffle(
                    ids
                )

                mask = np.zeros(
                    n,
                    dtype=bool,
                )

                mask[
                    ids[:k]
                ] = True

                random_metrics.append(
                    evaluate_mask(
                        g,
                        mask,
                    )
                )

            avg = {
                key: float(
                    np.mean(
                        [
                            x[key]
                            for x
                            in random_metrics
                        ]
                    )
                )
                for key
                in random_metrics[0]
            }

            rows.append({
                "case_id": case_id,
                "router": "Random",
                "budget": ratio,
                **avg,
            })

    case_df = pd.DataFrame(
        rows
    )

    case_df.to_csv(
        OUT_DIR
        / "budget_case_metrics.csv",
        index=False,
    )

    summary = (
        case_df.groupby(
            [
                "router",
                "budget",
            ]
        )
        .agg(
            ndcg10=(
                "ndcg10",
                "mean",
            ),
            hr10=(
                "hr10",
                "mean",
            ),
            mrr=(
                "mrr",
                "mean",
            ),
            rank=(
                "rank",
                "mean",
            ),
            pairwise_win_rate=(
                "pairwise_win_rate",
                "mean",
            ),
            pairwise_margin=(
                "pairwise_margin",
                "mean",
            ),
            n=(
                "case_id",
                "count",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT_DIR
        / "budget_curve.csv",
        index=False,
    )

    print(
        "\n===== BEST NDCG BY ROUTER ====="
    )

    best = (
        summary.sort_values(
            "ndcg10",
            ascending=False,
        )
        .groupby(
            "router",
            as_index=False,
        )
        .first()
        .sort_values(
            "ndcg10",
            ascending=False,
        )
    )

    print(
        best.to_string(
            index=False
        )
    )

    for metric in [
        "ndcg10",
        "pairwise_win_rate",
    ]:
        plt.figure(
            figsize=(8, 5)
        )

        for router in [
            "BenefitRF",
            "BenefitLR",
            "Random",
            "Tail",
            "LowKnowledge",
            "LabelOracle",
        ]:
            x = summary[
                summary["router"]
                == router
            ].sort_values(
                "budget"
            )

            plt.plot(
                x["budget"],
                x[metric],
                marker="o",
                label=router,
            )

        noaug = summary[
            summary["router"]
            == "NoAugment"
        ][metric].iloc[0]

        plt.axhline(
            noaug,
            linestyle="--",
            label="NoAugment",
        )

        plt.xlabel(
            "Augmentation budget ratio"
        )

        plt.ylabel(
            metric
        )

        plt.title(
            f"{metric} vs augmentation budget"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR
            / f"budget_{metric}.png",
            dpi=180,
        )

        plt.close()

    print(
        "\nSaved:",
        OUT_DIR
        / "budget_curve.csv"
    )


if __name__ == "__main__":
    main()
