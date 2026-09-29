import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src import config
from src.utility.train_ranker import FEATURES
from src.utility.evaluate_ranker import expected_tie_metrics


# ============================================================
# Fixed experimental protocol
# ============================================================

SEED = int(
    os.environ.get(
        "AUGVAL_SEED",
        "2027",
    )
)

DEVICE_NAME = os.environ.get(
    "AUGVAL_DEVICE",
    "cpu",
)

DEVICE = torch.device(
    DEVICE_NAME
)

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

MAX_EPOCHS = 100
PATIENCE = 12

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.10

PAIRWISE_WEIGHT = 0.10
UTILITY_WEIGHT_ALPHA = 2.0

EPS = 1e-8


OUT = (
    config.RESULT_DIR
    / "method_setrank_v2"
    / f"seed_{SEED}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Feature preprocessing
# ============================================================

def fit_feature_stats(
    train,
    features,
):

    x = train[
        features
    ].astype(float)

    median = x.median(
        axis=0
    )

    x = x.fillna(
        median
    )

    mean = x.mean(
        axis=0
    )

    std = x.std(
        axis=0
    )

    std = std.replace(
        0.0,
        1.0,
    )

    std = std.fillna(
        1.0
    )

    return (
        median,
        mean,
        std,
    )


def transform_features(
    df,
    features,
    median,
    mean,
    std,
):

    x = (
        df[
            features
        ]
        .astype(float)
        .fillna(
            median
        )
    )

    x = (
        x
        - mean
    ) / std

    return x.astype(
        np.float32
    )


# ============================================================
# Case-level dataset
# ============================================================

class CaseDataset(Dataset):

    def __init__(
        self,
        frame,
        feature_matrix,
        target_scale,
    ):

        self.samples = []

        for case_id, g in frame.groupby(
            "case_id",
            sort=False,
        ):

            idx = (
                g.index
                .to_numpy(
                    dtype=np.int64
                )
            )

            x = (
                feature_matrix
                .loc[idx]
                .to_numpy(
                    dtype=np.float32
                )
            )

            y = (
                g[
                    "utility_ndcg_if_aug"
                ]
                .to_numpy(
                    dtype=np.float32
                )
                /
                target_scale
            )

            self.samples.append(
                (
                    int(case_id),
                    idx,
                    x,
                    y,
                )
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        return self.samples[
            index
        ]


def collate_cases(batch):

    batch_size = len(batch)

    max_n = max(
        x[2].shape[0]
        for x in batch
    )

    feat_dim = (
        batch[0][2].shape[1]
    )

    features = torch.zeros(
        (
            batch_size,
            max_n,
            feat_dim,
        ),
        dtype=torch.float32,
    )

    targets = torch.zeros(
        (
            batch_size,
            max_n,
        ),
        dtype=torch.float32,
    )

    mask = torch.zeros(
        (
            batch_size,
            max_n,
        ),
        dtype=torch.bool,
    )

    row_indices = torch.full(
        (
            batch_size,
            max_n,
        ),
        -1,
        dtype=torch.long,
    )

    case_ids = []

    for b, (
        case_id,
        idx,
        x,
        y,
    ) in enumerate(batch):

        n = len(idx)

        features[
            b,
            :n,
        ] = torch.from_numpy(
            x
        )

        targets[
            b,
            :n,
        ] = torch.from_numpy(
            y
        )

        mask[
            b,
            :n,
        ] = True

        row_indices[
            b,
            :n,
        ] = torch.from_numpy(
            idx
        )

        case_ids.append(
            case_id
        )

    return {
        "case_ids":
            case_ids,

        "x":
            features,

        "y":
            targets,

        "mask":
            mask,

        "row_indices":
            row_indices,
    }


# ============================================================
# Models
# ============================================================

class PointwiseMLP(nn.Module):

    def __init__(
        self,
        input_dim,
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                HIDDEN_DIM,
            ),
            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),
            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                1,
            ),
        )

    def forward(
        self,
        x,
        mask,
    ):

        return (
            self.net(x)
            .squeeze(-1)
        )


class SetRankImpact(nn.Module):

    def __init__(
        self,
        input_dim,
    ):

        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                HIDDEN_DIM,
            ),
            nn.GELU(),
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM,
                nhead=N_HEADS,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=N_LAYERS,
            )
        )

        self.head = nn.Sequential(
            nn.Linear(
                HIDDEN_DIM * 2,
                HIDDEN_DIM,
            ),
            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                1,
            ),
        )

    def forward(
        self,
        x,
        mask,
    ):

        local = self.input_proj(
            x
        )

        context = self.encoder(
            local,
            src_key_padding_mask=(
                ~mask
            ),
        )

        joint = torch.cat(
            [
                local,
                context,
            ],
            dim=-1,
        )

        return (
            self.head(
                joint
            )
            .squeeze(-1)
        )


# ============================================================
# Loss
# ============================================================

def utility_loss(
    prediction,
    target,
    mask,
):

    valid_pred = (
        prediction[
            mask
        ]
    )

    valid_target = (
        target[
            mask
        ]
    )

    weights = (
        1.0
        +
        UTILITY_WEIGHT_ALPHA
        * torch.abs(
            valid_target
        )
    )

    pointwise = F.smooth_l1_loss(
        valid_pred,
        valid_target,
        reduction="none",
    )

    pointwise = (
        pointwise
        * weights
    ).mean()

    pair_losses = []

    batch_size = (
        prediction.shape[0]
    )

    for b in range(
        batch_size
    ):

        valid = mask[b]

        p = prediction[
            b,
            valid,
        ]

        y = target[
            b,
            valid,
        ]

        if len(p) < 2:
            continue

        dy = (
            y[:, None]
            -
            y[None, :]
        )

        dp = (
            p[:, None]
            -
            p[None, :]
        )

        upper = torch.triu(
            torch.ones_like(
                dy,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        informative = (
            upper
            &
            (
                torch.abs(dy)
                > EPS
            )
        )

        if not informative.any():
            continue

        direction = torch.sign(
            dy[
                informative
            ]
        )

        pred_diff = (
            dp[
                informative
            ]
        )

        importance = (
            torch.abs(
                dy[
                    informative
                ]
            )
        )

        importance = (
            importance
            /
            (
                importance.mean()
                + EPS
            )
        )

        pair = F.softplus(
            -direction
            * pred_diff
        )

        pair_losses.append(
            (
                pair
                * importance
            ).mean()
        )

    if pair_losses:

        pairwise = torch.stack(
            pair_losses
        ).mean()

    else:

        pairwise = torch.zeros(
            (),
            device=prediction.device,
        )

    total = (
        pointwise
        +
        PAIRWISE_WEIGHT
        * pairwise
    )

    return (
        total,
        pointwise.detach(),
        pairwise.detach(),
    )


# ============================================================
# Training
# ============================================================

def make_loader(
    frame,
    feature_matrix,
    target_scale,
    shuffle,
    seed,
):

    dataset = CaseDataset(
        frame,
        feature_matrix,
        target_scale,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_cases,
        num_workers=0,
        generator=generator,
    )


def epoch_loss(
    model,
    loader,
    optimizer=None,
):

    training = (
        optimizer
        is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    losses = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            x = batch[
                "x"
            ].to(
                DEVICE
            )

            y = batch[
                "y"
            ].to(
                DEVICE
            )

            mask = batch[
                "mask"
            ].to(
                DEVICE
            )

            pred = model(
                x,
                mask,
            )

            loss, _, _ = utility_loss(
                pred,
                y,
                mask,
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

    return float(
        np.mean(
            losses
        )
    )


def find_best_epoch(
    model_class,
    input_dim,
    inner_train,
    inner_es,
    feature_matrix,
    target_scale,
    seed,
):

    set_seed(seed)

    model = model_class(
        input_dim
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(
        inner_train,
        feature_matrix,
        target_scale,
        True,
        seed,
    )

    es_loader = make_loader(
        inner_es,
        feature_matrix,
        target_scale,
        False,
        seed,
    )

    best_loss = float(
        "inf"
    )

    best_epoch = 1
    stale = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = epoch_loss(
            model,
            train_loader,
            optimizer,
        )

        es_loss = epoch_loss(
            model,
            es_loader,
            None,
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "earlystop_loss":
                es_loss,
        })

        if (
            es_loss
            < best_loss
            - 1e-6
        ):

            best_loss = (
                es_loss
            )

            best_epoch = (
                epoch
            )

            stale = 0

        else:

            stale += 1

        if stale >= PATIENCE:
            break

    return (
        best_epoch,
        pd.DataFrame(
            history
        ),
    )


def fit_full_train(
    model_class,
    input_dim,
    train,
    feature_matrix,
    target_scale,
    epochs,
    seed,
):

    set_seed(seed)

    model = model_class(
        input_dim
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loader = make_loader(
        train,
        feature_matrix,
        target_scale,
        True,
        seed,
    )

    for _ in range(
        epochs
    ):

        epoch_loss(
            model,
            loader,
            optimizer,
        )

    return model


def predict_rows(
    model,
    frame,
    feature_matrix,
    target_scale,
):

    loader = make_loader(
        frame,
        feature_matrix,
        target_scale,
        False,
        SEED,
    )

    prediction = np.zeros(
        len(frame),
        dtype=float,
    )

    model.eval()

    with torch.no_grad():

        for batch in loader:

            x = batch[
                "x"
            ].to(
                DEVICE
            )

            mask = batch[
                "mask"
            ].to(
                DEVICE
            )

            pred = (
                model(
                    x,
                    mask,
                )
                .cpu()
                .numpy()
            )

            indices = (
                batch[
                    "row_indices"
                ]
                .numpy()
            )

            mask_np = (
                batch[
                    "mask"
                ]
                .numpy()
            )

            for b in range(
                len(indices)
            ):

                valid_idx = (
                    indices[
                        b,
                        mask_np[b],
                    ]
                )

                valid_pred = (
                    pred[
                        b,
                        mask_np[b],
                    ]
                    * target_scale
                )

                prediction[
                    valid_idx
                ] = valid_pred

    return prediction


# ============================================================
# Recommendation evaluation
# ============================================================

def evaluate_policy(
    frame,
    prediction,
    budget,
    abstain,
):

    work = frame.copy()

    work[
        "_prediction"
    ] = prediction

    rows = []

    for case_id, g in work.groupby(
        "case_id",
        sort=False,
    ):

        g = g.reset_index(
            drop=True
        )

        pred = (
            g[
                "_prediction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        n = len(g)

        k = max(
            1,
            int(
                round(
                    budget * n
                )
            ),
        )

        order = np.argsort(
            -pred,
            kind="stable",
        )

        if abstain:

            order = order[
                pred[
                    order
                ] > 0
            ]

        selected = (
            order[
                :k
            ]
        )

        base = (
            g[
                "base_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        clean = (
            g[
                "clean_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        labels = (
            g[
                "is_target"
            ]
            .to_numpy(
                dtype=int
            )
        )

        final_scores = (
            base.copy()
        )

        final_scores[
            selected
        ] = clean[
            selected
        ]

        metrics = (
            expected_tie_metrics(
                final_scores,
                labels,
            )
        )

        true_utility = (
            g[
                "utility_ndcg_if_aug"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(selected):

            selected_utility = (
                true_utility[
                    selected
                ]
            )

            harm = float(
                (
                    selected_utility
                    < -EPS
                ).mean()
            )

            help_ = float(
                (
                    selected_utility
                    > EPS
                ).mean()
            )

        else:

            harm = 0.0
            help_ = 0.0

        rows.append({
            "case_id":
                int(case_id),

            "ndcg10":
                metrics[
                    "ndcg10"
                ],

            "hr10":
                metrics[
                    "hr10"
                ],

            "mrr":
                metrics[
                    "mrr"
                ],

            "pairwise_win_rate":
                metrics[
                    "pairwise_win_rate"
                ],

            "augment_ratio":
                (
                    len(selected)
                    / n
                ),

            "selected_harm_rate":
                harm,

            "selected_help_rate":
                help_,
        })

    return pd.DataFrame(
        rows
    )


def summarize(
    cases,
):

    return {
        "ndcg10":
            float(
                cases[
                    "ndcg10"
                ].mean()
            ),

        "hr10":
            float(
                cases[
                    "hr10"
                ].mean()
            ),

        "mrr":
            float(
                cases[
                    "mrr"
                ].mean()
            ),

        "pairwise_win_rate":
            float(
                cases[
                    "pairwise_win_rate"
                ].mean()
            ),

        "augment_ratio":
            float(
                cases[
                    "augment_ratio"
                ].mean()
            ),

        "selected_harm_rate":
            float(
                cases[
                    "selected_harm_rate"
                ].mean()
            ),

        "selected_help_rate":
            float(
                cases[
                    "selected_help_rate"
                ].mean()
            ),
    }


def evaluate_curve(
    frame,
    prediction,
    method,
    abstain,
):

    rows = []

    for budget in BUDGETS:

        cases = evaluate_policy(
            frame,
            prediction,
            budget,
            abstain,
        )

        s = summarize(
            cases
        )

        rows.append({
            "method":
                method,

            "budget":
                budget,

            **s,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )

    benefit_path = (
        config.RESULT_DIR
        / "benefit"
        / "benefit_dataset.csv"
    )

    split_path = (
        config.DATA_DIR
        / "formal"
        / "split_manifest.csv"
    )

    if not benefit_path.exists():
        raise FileNotFoundError(
            benefit_path
        )

    if not split_path.exists():
        raise FileNotFoundError(
            split_path
        )

    df = pd.read_csv(
        benefit_path
    )

    split = pd.read_csv(
        split_path
    )

    df[
        "user_id"
    ] = (
        df[
            "user_id"
        ]
        .astype(str)
    )

    split[
        "user_id"
    ] = (
        split[
            "user_id"
        ]
        .astype(str)
    )

    df = df.merge(
        split,
        on=[
            "case_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    required = {
        "case_id",
        "user_id",
        "base_score",
        "clean_score",
        "is_target",
        "utility_ndcg_if_aug",
        "split",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: "
            f"{missing}"
        )

    if df[
        "split"
    ].isna().any():

        raise RuntimeError(
            "Missing split labels."
        )

    features = [
        f
        for f in FEATURES
        if f in df.columns
    ]

    forbidden = {
        "is_target",
        "clean_score",
        "benefit_label",
        "utility_ndcg_if_aug",
        "utility_pwr_if_aug",
        "utility_margin_if_aug",
    }

    leakage = (
        set(features)
        & forbidden
    )

    if leakage:

        raise RuntimeError(
            f"Feature leakage: "
            f"{leakage}"
        )

    train = (
        df[
            df[
                "split"
            ] == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val = (
        df[
            df[
                "split"
            ] == "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Make row index explicit because prediction is restored by row.
    train.index = np.arange(
        len(train)
    )

    val.index = np.arange(
        len(val)
    )

    median, mean, std = (
        fit_feature_stats(
            train,
            features,
        )
    )

    train_x = transform_features(
        train,
        features,
        median,
        mean,
        std,
    )

    val_x = transform_features(
        val,
        features,
        median,
        mean,
        std,
    )

    target_std = float(
        train[
            "utility_ndcg_if_aug"
        ].std()
    )

    target_scale = max(
        target_std,
        1e-4,
    )

    # ========================================================
    # Inner train / early-stop split by CASE, not candidate.
    # ========================================================

    train_cases = (
        train[
            "case_id"
        ]
        .drop_duplicates()
        .to_numpy(
            copy=True
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(
        train_cases
    )

    n_es = max(
        1,
        int(
            round(
                0.10
                * len(
                    train_cases
                )
            )
        ),
    )

    es_cases = set(
        train_cases[
            :n_es
        ]
    )

    inner_es = (
        train[
            train[
                "case_id"
            ].isin(
                es_cases
            )
        ]
        .copy()
    )

    inner_train = (
        train[
            ~train[
                "case_id"
            ].isin(
                es_cases
            )
        ]
        .copy()
    )

    print(
        "============================================================"
    )

    print(
        "SET RANK IMPACT V2"
    )

    print(
        "dataset =",
        config.DATASET_NAME
    )

    print(
        "model   =",
        config.MODEL_NAME
    )

    print(
        "seed    =",
        SEED
    )

    print(
        "device  =",
        DEVICE
    )

    print(
        "features =",
        len(features)
    )

    print(
        "train cases =",
        train[
            "case_id"
        ].nunique()
    )

    print(
        "inner train cases =",
        inner_train[
            "case_id"
        ].nunique()
    )

    print(
        "early-stop cases =",
        inner_es[
            "case_id"
        ].nunique()
    )

    print(
        "formal val cases =",
        val[
            "case_id"
        ].nunique()
    )

    print(
        "target scale =",
        target_scale
    )

    print(
        "============================================================"
    )

    models = [
        (
            "PointwiseMLP",
            PointwiseMLP,
        ),
        (
            "SetRankImpact",
            SetRankImpact,
        ),
    ]

    all_curves = []
    diagnostics = []

    val_predictions = (
        val[
            [
                "case_id",
                "user_id",
            ]
        ]
        .copy()
    )

    val_predictions[
        "true_utility"
    ] = (
        val[
            "utility_ndcg_if_aug"
        ]
        .to_numpy()
    )

    for model_name, model_class in models:

        print()
        print(
            "===== ",
            model_name,
            " =====",
            sep="",
        )

        best_epoch, history = (
            find_best_epoch(
                model_class,
                len(features),
                inner_train,
                inner_es,
                train_x,
                target_scale,
                SEED,
            )
        )

        history.to_csv(
            OUT
            / (
                f"{model_name}_"
                f"training_history.csv"
            ),
            index=False,
        )

        print(
            "best epoch =",
            best_epoch
        )

        model = fit_full_train(
            model_class,
            len(features),
            train,
            train_x,
            target_scale,
            best_epoch,
            SEED,
        )

        pred = predict_rows(
            model,
            val,
            val_x,
            target_scale,
        )

        val_predictions[
            model_name
        ] = pred

        truth = (
            val[
                "utility_ndcg_if_aug"
            ]
            .to_numpy(
                dtype=float
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    pred
                    - truth
                )
            )
        )

        pred_rank = (
            pd.Series(pred)
            .rank()
            .to_numpy()
        )

        true_rank = (
            pd.Series(truth)
            .rank()
            .to_numpy()
        )

        spearman = float(
            np.corrcoef(
                pred_rank,
                true_rank,
            )[0, 1]
        )

        diagnostics.append({
            "method":
                model_name,

            "best_epoch":
                best_epoch,

            "mae":
                mae,

            "spearman":
                spearman,
        })

        exact_curve = (
            evaluate_curve(
                val,
                pred,
                model_name,
                False,
            )
        )

        abstain_curve = (
            evaluate_curve(
                val,
                pred,
                (
                    model_name
                    + "Abstain"
                ),
                True,
            )
        )

        all_curves.extend([
            exact_curve,
            abstain_curve,
        ])

    curve = pd.concat(
        all_curves,
        ignore_index=True,
    )

    curve[
        "dataset"
    ] = config.DATASET_NAME

    curve[
        "model"
    ] = config.MODEL_NAME

    curve[
        "seed"
    ] = SEED

    curve.to_csv(
        OUT
        / "validation_budget_curve.csv",
        index=False,
    )

    selected_rows = []

    for method, g in curve.groupby(
        "method",
        sort=False,
    ):

        best = (
            g.sort_values(
                [
                    "ndcg10",
                    "augment_ratio",
                    "budget",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        selected_rows.append(
            best
        )

    selected = pd.DataFrame(
        selected_rows
    )

    selected.to_csv(
        OUT
        / "validation_selected.csv",
        index=False,
    )

    diagnostics = pd.DataFrame(
        diagnostics
    )

    diagnostics.to_csv(
        OUT
        / "prediction_diagnostics.csv",
        index=False,
    )

    val_predictions.to_csv(
        OUT
        / "validation_predictions.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

        "seed":
            SEED,

        "device":
            str(DEVICE),

        "features":
            features,

        "budgets":
            BUDGETS,

        "hidden_dim":
            HIDDEN_DIM,

        "n_heads":
            N_HEADS,

        "n_layers":
            N_LAYERS,

        "ff_dim":
            FF_DIM,

        "dropout":
            DROPOUT,

        "pairwise_weight":
            PAIRWISE_WEIGHT,

        "utility_weight_alpha":
            UTILITY_WEIGHT_ALPHA,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "test_evaluated":
            False,
    }

    with (
        OUT
        / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "===== PREDICTION DIAGNOSTICS ====="
    )

    print(
        diagnostics.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "===== VALIDATION SELECTED ====="
    )

    print(
        selected[
            [
                "method",
                "budget",
                "ndcg10",
                "hr10",
                "mrr",
                "pairwise_win_rate",
                "augment_ratio",
                "selected_harm_rate",
                "selected_help_rate",
            ]
        ]
        .sort_values(
            "ndcg10",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "Saved:",
        OUT
    )


if __name__ == "__main__":
    main()
