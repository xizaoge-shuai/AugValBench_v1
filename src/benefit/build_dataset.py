import json
import math

import numpy as np
import pandas as pd

from src import config
from src.utils import read_jsonl, dcg_single_positive


OUT_DIR = config.RESULT_DIR / "benefit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ranking_metrics(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    target_indices = np.where(labels == 1)[0]

    if len(target_indices) != 1:
        raise ValueError("Each case must contain exactly one target.")

    target_idx = int(target_indices[0])

    order = np.argsort(-scores, kind="stable")
    rank = int(np.where(order == target_idx)[0][0]) + 1

    target_score = scores[target_idx]
    neg_scores = scores[labels == 0]

    wins = (target_score > neg_scores).astype(float)
    ties = (target_score == neg_scores).astype(float)

    pwr = float(np.mean(wins + 0.5 * ties))

    margin = float(
        target_score - np.mean(neg_scores)
    )

    ndcg = dcg_single_positive(
        rank,
        10,
    )

    return {
        "rank": rank,
        "ndcg10": ndcg,
        "pairwise_win_rate": pwr,
        "pairwise_margin": margin,
    }


def main():
    scores = pd.read_csv(
        config.RESULT_DIR / "rerank_scores.csv"
    )

    scores["item_id"] = scores["item_id"].astype(str)

    base = scores[
        scores["condition"] == "base"
    ][
        [
            "case_id",
            "user_id",
            "item_id",
            "is_target",
            "score",
        ]
    ].rename(
        columns={
            "score": "base_score",
        }
    )

    clean = scores[
        scores["condition"] == "clean"
    ][
        [
            "case_id",
            "item_id",
            "score",
        ]
    ].rename(
        columns={
            "score": "clean_score",
        }
    )

    df = base.merge(
        clean,
        on=[
            "case_id",
            "item_id",
        ],
        how="inner",
    )

    # ---------- Knowledge features ----------
    knowledge_path = (
        config.RESULT_DIR
        / "item_knowledge.csv"
    )

    if knowledge_path.exists():
        knowledge = pd.read_csv(
            knowledge_path
        )

        knowledge["item_id"] = (
            knowledge["item_id"]
            .astype(str)
        )

        keep = [
            x for x in [
                "item_id",
                "knowledge_score",
                "probe_accuracy",
                "n_probes",
                "mean_co_count",
                "mean_association_score",
                "mean_semantic_margin",
                "mean_abs_position_bias",
            ]
            if x in knowledge.columns
        ]

        df = df.merge(
            knowledge[keep],
            on="item_id",
            how="left",
        )

    if "knowledge_score" not in df:
        df["knowledge_score"] = np.nan

    df["knowledge_missing"] = (
        df["knowledge_score"]
        .isna()
        .astype(int)
    )

    # ---------- Popularity ----------
    with (
        config.DATA_DIR
        / "item_popularity.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        popularity = {
            str(k): float(v)
            for k, v in json.load(f).items()
        }

    df["item_popularity"] = (
        df["item_id"]
        .map(popularity)
        .fillna(0.0)
    )

    df["item_popularity_log"] = np.log1p(
        df["item_popularity"]
    )

    # ---------- User history ----------
    eval_cases = read_jsonl(
        config.DATA_DIR
        / "eval_cases.jsonl"
    )

    case_features = {}

    for c in eval_cases:
        history = [
            str(x)
            for x in c["history"]
        ]

        hp = np.array(
            [
                popularity.get(x, 0.0)
                for x in history
            ],
            dtype=float,
        )

        case_features[int(c["case_id"])] = {
            "history_len": len(history),
            "history_pop_mean_log": (
                float(
                    np.log1p(hp).mean()
                )
                if len(hp)
                else 0.0
            ),
            "history_pop_std_log": (
                float(
                    np.log1p(hp).std()
                )
                if len(hp)
                else 0.0
            ),
        }

    case_feature_df = (
        pd.DataFrame.from_dict(
            case_features,
            orient="index",
        )
        .reset_index()
        .rename(
            columns={
                "index": "case_id",
            }
        )
    )

    df = df.merge(
        case_feature_df,
        on="case_id",
        how="left",
    )

    # ---------- Metadata quality ----------
    with (
        config.DATA_DIR
        / "item_meta.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        meta = json.load(f)

    def meta_feature(item_id):
        m = meta.get(
            str(item_id),
            {},
        )

        title = str(
            m.get("title", "")
        )
        desc = str(
            m.get("description", "")
        )
        feat = str(
            m.get("features", "")
        )
        cat = str(
            m.get("main_category", "")
        )
        store = str(
            m.get("store", "")
        )

        return (
            len(title),
            len(desc),
            len(feat),
            len(title)
            + len(desc)
            + len(feat),
            int(bool(cat.strip())),
            int(bool(store.strip())),
        )

    meta_features = df[
        "item_id"
    ].map(
        meta_feature
    )

    df[
        [
            "title_len",
            "description_len",
            "features_len",
            "meta_total_len",
            "category_present",
            "store_present",
        ]
    ] = pd.DataFrame(
        meta_features.tolist(),
        index=df.index,
    )

    # ---------- Base ranking context ----------
    extra_rows = []

    for case_id, g in df.groupby(
        "case_id",
        sort=False,
    ):
        g = g.copy()

        base_scores = (
            g["base_score"]
            .to_numpy(dtype=float)
        )

        labels = (
            g["is_target"]
            .to_numpy(dtype=int)
        )

        clean_scores = (
            g["clean_score"]
            .to_numpy(dtype=float)
        )

        base_metric = ranking_metrics(
            base_scores,
            labels,
        )

        mean_score = float(
            np.mean(base_scores)
        )

        std_score = float(
            np.std(base_scores)
        )

        max_score = float(
            np.max(base_scores)
        )

        # rank feature
        rank_series = (
            pd.Series(base_scores)
            .rank(
                ascending=False,
                method="average",
            )
            .to_numpy()
        )

        for local_idx, (
            index,
            row,
        ) in enumerate(
            g.iterrows()
        ):
            hybrid = base_scores.copy()
            hybrid[local_idx] = (
                clean_scores[local_idx]
            )

            after = ranking_metrics(
                hybrid,
                labels,
            )

            delta_ndcg = (
                after["ndcg10"]
                - base_metric["ndcg10"]
            )

            delta_pwr = (
                after["pairwise_win_rate"]
                - base_metric["pairwise_win_rate"]
            )

            delta_margin = (
                after["pairwise_margin"]
                - base_metric["pairwise_margin"]
            )

            base_score = float(
                base_scores[local_idx]
            )

            z = (
                (base_score - mean_score)
                / std_score
                if std_score > 1e-12
                else 0.0
            )

            extra_rows.append({
                "index": index,

                "base_rank": float(
                    rank_series[local_idx]
                ),

                "base_rank_norm": float(
                    (
                        rank_series[local_idx] - 1
                    )
                    / max(
                        1,
                        len(g) - 1,
                    )
                ),

                "base_score_z": z,

                "base_top_gap": (
                    max_score
                    - base_score
                ),

                "case_base_score_mean": (
                    mean_score
                ),

                "case_base_score_std": (
                    std_score
                ),

                "utility_ndcg_if_aug": (
                    delta_ndcg
                ),

                "utility_pwr_if_aug": (
                    delta_pwr
                ),

                "utility_margin_if_aug": (
                    delta_margin
                ),

                # 第一版预测目标：
                # augmentation 是否使该 candidate
                # 朝“正确排序方向”移动。
                "benefit_label": int(
                    delta_margin > 0
                ),
            })

    extra = (
        pd.DataFrame(extra_rows)
        .set_index("index")
    )

    df = df.join(extra)

    # clean score 只用于生成 label，
    # 后续训练时绝不能作为 feature。
    out_path = (
        OUT_DIR
        / "benefit_dataset.csv"
    )

    df.to_csv(
        out_path,
        index=False,
    )

    print(
        "\n===== BENEFIT DATASET ====="
    )
    print(
        "rows =",
        len(df),
    )
    print(
        "cases =",
        df["case_id"].nunique(),
    )

    print(
        "\n===== LABEL ====="
    )
    print(
        df["benefit_label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\npositive rate =",
        df["benefit_label"].mean(),
    )

    print(
        "\n===== COUNTERFACTUAL UTILITY ====="
    )

    print(
        df[
            [
                "utility_ndcg_if_aug",
                "utility_pwr_if_aug",
                "utility_margin_if_aug",
            ]
        ]
        .describe(
            percentiles=[
                .01,
                .05,
                .10,
                .25,
                .50,
                .75,
                .90,
                .95,
                .99,
            ]
        )
        .to_string()
    )

    print(
        "\nSaved:",
        out_path,
    )


if __name__ == "__main__":
    main()
