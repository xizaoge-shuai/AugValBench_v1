import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from src import config


def corr_line(name, x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 5:
        return f"{name}: insufficient data (n={len(tmp)})"
    rho, p = spearmanr(tmp["x"], tmp["y"])
    return f"{name}: rho={rho:.4f}, p={p:.4g}, n={len(tmp)}"


def main():
    metrics = pd.read_csv(config.RESULT_DIR / "case_metrics.csv")
    scores = pd.read_csv(config.RESULT_DIR / "rerank_scores.csv")

    base = metrics[metrics["condition"] == "base"][
        [
            "case_id",
            "ndcg10",
            "hr10",
            "mrr",
            "target_rank",
            "target_score",
            "pairwise_win_rate",
            "pairwise_margin",
        ]
    ].rename(columns={
        "ndcg10": "base_ndcg10",
        "hr10": "base_hr10",
        "mrr": "base_mrr",
        "target_rank": "base_rank",
        "target_score": "base_target_score",
        "pairwise_win_rate": "base_pairwise_win_rate",
        "pairwise_margin": "base_pairwise_margin",
    })

    gain = metrics.merge(base, on="case_id", how="left")
    gain["delta_ndcg10"] = gain["ndcg10"] - gain["base_ndcg10"]
    gain["delta_hr10"] = gain["hr10"] - gain["base_hr10"]
    gain["delta_mrr"] = gain["mrr"] - gain["base_mrr"]
    gain["delta_target_score"] = gain["target_score"] - gain["base_target_score"]
    gain["delta_pairwise_win_rate"] = (
        gain["pairwise_win_rate"] - gain["base_pairwise_win_rate"]
    )
    gain["delta_pairwise_margin"] = (
        gain["pairwise_margin"] - gain["base_pairwise_margin"]
    )
    gain.to_csv(config.RESULT_DIR / "gain_table.csv", index=False)

    summary = gain.groupby("condition").agg(
        ndcg10=("ndcg10", "mean"),
        delta_ndcg10=("delta_ndcg10", "mean"),
        delta_pairwise_win_rate=("delta_pairwise_win_rate", "mean"),
        delta_pairwise_margin=("delta_pairwise_margin", "mean"),
        harmful_rate=("delta_ndcg10", lambda s: float((s < 0).mean())),
        helpful_rate=("delta_ndcg10", lambda s: float((s > 0).mean())),
        pairwise_harmful_rate=(
            "delta_pairwise_win_rate",
            lambda s: float((s < 0).mean()),
        ),
        pairwise_helpful_rate=(
            "delta_pairwise_win_rate",
            lambda s: float((s > 0).mean()),
        ),
        n=("case_id", "count"),
    ).reset_index()

    summary.to_csv(config.RESULT_DIR / "summary_by_condition.csv", index=False)
    print("\n=== condition summary ===")
    print(summary.to_string(index=False))

    target = scores[scores["is_target"] == 1][
        ["case_id", "item_id"]
    ].drop_duplicates()
    target["item_id"] = target["item_id"].astype(str)

    clean = gain[gain["condition"] == "clean"].merge(
        target,
        on="case_id",
        how="left",
    )

    lines = []

    kb_path = config.RESULT_DIR / "item_knowledge.csv"
    if kb_path.exists():
        kb = pd.read_csv(kb_path)
        kb["item_id"] = kb["item_id"].astype(str)

        clean = clean.merge(
            kb[[
                "item_id",
                "knowledge_score",
                "mean_abs_position_bias",
            ]],
            on="item_id",
            how="left",
        )

        lines.append(
            corr_line(
                "Spearman(knowledge_score, delta_ndcg10)",
                clean["knowledge_score"],
                clean["delta_ndcg10"],
            )
        )
        lines.append(
            corr_line(
                "Spearman(knowledge_score, delta_pairwise_win_rate)",
                clean["knowledge_score"],
                clean["delta_pairwise_win_rate"],
            )
        )
        lines.append(
            corr_line(
                "Spearman(knowledge_score, delta_pairwise_margin)",
                clean["knowledge_score"],
                clean["delta_pairwise_margin"],
            )
        )

        tmp = clean.dropna(
            subset=["knowledge_score", "delta_pairwise_win_rate"]
        )
        if len(tmp) >= 5:
            plt.figure(figsize=(6.5, 5))
            plt.scatter(
                tmp["knowledge_score"],
                tmp["delta_pairwise_win_rate"],
                alpha=0.65,
            )
            plt.axhline(0, linewidth=1)
            plt.xlabel("Position-debiased knowledge score")
            plt.ylabel("Clean augmentation gain: ΔPairwise Win Rate")
            plt.title("Knowledge Gap vs Actual Augmentation Gain")
            plt.tight_layout()
            plt.savefig(config.PLOT_DIR / "knowledge_vs_gain.png", dpi=180)
            plt.close()

    with (config.DATA_DIR / "item_popularity.json").open("r", encoding="utf-8") as f:
        popularity = {str(k): float(v) for k, v in json.load(f).items()}

    clean["popularity"] = clean["item_id"].map(popularity).fillna(0.0)
    clean["pop_bucket"] = pd.qcut(
        clean["popularity"].rank(method="first"),
        q=4,
        labels=["Q1-tail", "Q2", "Q3", "Q4-head"],
    )

    pop_summary = clean.groupby("pop_bucket", observed=True)[
        ["delta_ndcg10", "delta_pairwise_win_rate"]
    ].mean()
    pop_summary.to_csv(config.RESULT_DIR / "popularity_gain.csv")

    plt.figure(figsize=(6.5, 4.5))
    pop_summary["delta_pairwise_win_rate"].plot(kind="bar")
    plt.axhline(0, linewidth=1)
    plt.ylabel("Mean ΔPairwise Win Rate (clean - base)")
    plt.xlabel("Target popularity bucket")
    plt.title("Augmentation Gain by Popularity")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "popularity_gain.png", dpi=180)
    plt.close()

    plot_df = summary[summary["condition"] != "base"].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(plot_df["condition"], plot_df["delta_pairwise_win_rate"])
    plt.axhline(0, linewidth=1)
    plt.ylabel("Mean ΔPairwise Win Rate vs Base")
    plt.xlabel("Augmentation condition")
    plt.title("Controlled Augmentation Utility")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "condition_ndcg.png", dpi=180)
    plt.close()

    lines.append(
        "Clean NDCG-harmful rate: "
        f"{float((clean['delta_ndcg10'] < 0).mean()):.4f}"
    )
    lines.append(
        "Clean NDCG-helpful rate: "
        f"{float((clean['delta_ndcg10'] > 0).mean()):.4f}"
    )
    lines.append(
        "Clean pairwise-harmful rate: "
        f"{float((clean['delta_pairwise_win_rate'] < 0).mean()):.4f}"
    )
    lines.append(
        "Clean pairwise-helpful rate: "
        f"{float((clean['delta_pairwise_win_rate'] > 0).mean()):.4f}"
    )

    (config.RESULT_DIR / "knowledge_gain_stats.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== core diagnostic ===")
    print("\n".join(lines))
    print("\nplots:", config.PLOT_DIR)


if __name__ == "__main__":
    main()
