import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src import config
from src.utility.train_ranker import FEATURES
from src.utility.evaluate_ranker import expected_tie_metrics


SEED = int(
    os.environ.get(
        "AUGVAL_SEED",
        "2027",
    )
)

BUDGETS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

N_RANDOM_STATES_PER_SIZE = 2

EPS = 1e-12

OUT = (
    config.RESULT_DIR
    / "method_sequential_v32_gpu"
    / f"seed_{SEED}"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Basic helpers
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)


def ranking_metrics(
    scores,
    labels,
):

    return expected_tie_metrics(
        np.asarray(
            scores,
            dtype=float,
        ),
        np.asarray(
            labels,
            dtype=int,
        ),
    )


def ndcg(
    scores,
    labels,
):

    return float(
        ranking_metrics(
            scores,
            labels,
        )["ndcg10"]
    )


def make_xgb(seed):

    device = os.environ.get(
        "AUGVAL_DEVICE",
        "cuda",
    )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                min_child_weight=3.0,
                subsample=0.90,
                colsample_bytree=0.90,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                device=device,
                n_jobs=-1,
                random_state=seed,
                verbosity=1,
            ),
        ),
    ])


# ============================================================
# Dynamic state
#
# IMPORTANT:
# These are all available before augmenting the NEXT
# candidate.
#
# clean_score is used ONLY for candidates that have already
# been augmented, because their post-augmentation score is
# then observable.
#
# No target identity / relevance label / current NDCG is used.
# ============================================================

DYNAMIC_FEATURES = [
    "dyn_selected_count",
    "dyn_selected_frac",

    "dyn_current_rank",
    "dyn_current_rank_norm",
    "dyn_current_rr",

    "dyn_current_z",

    "dyn_gap_above",
    "dyn_gap_below",
    "dyn_nearest_gap",

    "dyn_gap_top1",
    "dyn_gap_top3",
    "dyn_gap_top5",
    "dyn_gap_top10",

    "dyn_current_mean",
    "dyn_current_std",
    "dyn_current_range",

    "dyn_shift_mean",
    "dyn_shift_std",
    "dyn_shift_min",
    "dyn_shift_max",
    "dyn_shift_abs_mean",
    "dyn_shift_sum",

    "dyn_shift_positive_frac",
    "dyn_shift_negative_frac",

    "dyn_gap_nearest_selected",
]


def build_action_features(
    case,
    selected,
    candidate_indices,
    base_features,
):

    base = (
        case[
            "base_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    clean = (
        case[
            "clean_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    n = len(case)

    current = base.copy()

    selected = list(
        selected
    )

    if selected:

        idx = np.asarray(
            selected,
            dtype=int,
        )

        current[
            idx
        ] = clean[
            idx
        ]

    ranks = (
        pd.Series(
            current
        )
        .rank(
            method="average",
            ascending=False,
        )
        .to_numpy(
            dtype=float
        )
    )

    order = np.argsort(
        -current,
        kind="stable",
    )

    sorted_scores = (
        current[
            order
        ]
    )

    gap_above = np.zeros(
        n,
        dtype=float,
    )

    gap_below = np.zeros(
        n,
        dtype=float,
    )

    score_range = float(
        np.max(current)
        -
        np.min(current)
    )

    for pos, idx in enumerate(
        order
    ):

        if pos == 0:

            gap_above[idx] = (
                score_range
            )

        else:

            gap_above[idx] = (
                sorted_scores[
                    pos - 1
                ]
                -
                sorted_scores[
                    pos
                ]
            )

        if pos == n - 1:

            gap_below[idx] = (
                score_range
            )

        else:

            gap_below[idx] = (
                sorted_scores[
                    pos
                ]
                -
                sorted_scores[
                    pos + 1
                ]
            )

    nearest_gap = np.minimum(
        gap_above,
        gap_below,
    )

    def boundary(k):

        pos = min(
            k - 1,
            n - 1,
        )

        return float(
            sorted_scores[pos]
        )

    b1 = boundary(1)
    b3 = boundary(3)
    b5 = boundary(5)
    b10 = boundary(10)

    current_mean = float(
        np.mean(current)
    )

    current_std = float(
        np.std(current)
    )

    current_range = float(
        np.max(current)
        -
        np.min(current)
    )

    if current_std < EPS:
        current_std_safe = 1.0
    else:
        current_std_safe = (
            current_std
        )

    if selected:

        selected_idx = np.asarray(
            selected,
            dtype=int,
        )

        shifts = (
            clean[
                selected_idx
            ]
            -
            base[
                selected_idx
            ]
        )

        shift_mean = float(
            np.mean(shifts)
        )

        shift_std = float(
            np.std(shifts)
        )

        shift_min = float(
            np.min(shifts)
        )

        shift_max = float(
            np.max(shifts)
        )

        shift_abs_mean = float(
            np.mean(
                np.abs(shifts)
            )
        )

        shift_sum = float(
            np.sum(shifts)
        )

        shift_positive_frac = float(
            np.mean(
                shifts > EPS
            )
        )

        shift_negative_frac = float(
            np.mean(
                shifts < -EPS
            )
        )

        selected_current_scores = (
            current[
                selected_idx
            ]
        )

    else:

        shift_mean = 0.0
        shift_std = 0.0
        shift_min = 0.0
        shift_max = 0.0
        shift_abs_mean = 0.0
        shift_sum = 0.0

        shift_positive_frac = 0.0
        shift_negative_frac = 0.0

        selected_current_scores = None

    rows = []

    for j in candidate_indices:

        row = {}

        for feature in base_features:

            row[
                feature
            ] = case.iloc[j][
                feature
            ]

        if (
            selected_current_scores
            is None
        ):

            nearest_selected = (
                current_range
            )

        else:

            nearest_selected = float(
                np.min(
                    np.abs(
                        current[j]
                        -
                        selected_current_scores
                    )
                )
            )

        row.update({
            "dyn_selected_count":
                float(
                    len(selected)
                ),

            "dyn_selected_frac":
                float(
                    len(selected)
                    / max(
                        n,
                        1,
                    )
                ),

            "dyn_current_rank":
                float(
                    ranks[j]
                ),

            "dyn_current_rank_norm":
                float(
                    (
                        ranks[j] - 1.0
                    )
                    /
                    max(
                        n - 1,
                        1,
                    )
                ),

            "dyn_current_rr":
                float(
                    1.0
                    /
                    ranks[j]
                ),

            "dyn_current_z":
                float(
                    (
                        current[j]
                        -
                        current_mean
                    )
                    /
                    current_std_safe
                ),

            "dyn_gap_above":
                float(
                    gap_above[j]
                ),

            "dyn_gap_below":
                float(
                    gap_below[j]
                ),

            "dyn_nearest_gap":
                float(
                    nearest_gap[j]
                ),

            "dyn_gap_top1":
                float(
                    current[j]
                    -
                    b1
                ),

            "dyn_gap_top3":
                float(
                    current[j]
                    -
                    b3
                ),

            "dyn_gap_top5":
                float(
                    current[j]
                    -
                    b5
                ),

            "dyn_gap_top10":
                float(
                    current[j]
                    -
                    b10
                ),

            "dyn_current_mean":
                current_mean,

            "dyn_current_std":
                current_std,

            "dyn_current_range":
                current_range,

            "dyn_shift_mean":
                shift_mean,

            "dyn_shift_std":
                shift_std,

            "dyn_shift_min":
                shift_min,

            "dyn_shift_max":
                shift_max,

            "dyn_shift_abs_mean":
                shift_abs_mean,

            "dyn_shift_sum":
                shift_sum,

            "dyn_shift_positive_frac":
                shift_positive_frac,

            "dyn_shift_negative_frac":
                shift_negative_frac,

            "dyn_gap_nearest_selected":
                nearest_selected,
        })

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Counterfactual labels
# ============================================================

def current_scores(
    case,
    selected,
):

    base = (
        case[
            "base_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    clean = (
        case[
            "clean_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    scores = base.copy()

    if selected:

        idx = np.asarray(
            selected,
            dtype=int,
        )

        scores[
            idx
        ] = clean[
            idx
        ]

    return scores


def conditional_utilities(
    case,
    selected,
    candidate_indices,
):

    labels = (
        case[
            "is_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    clean = (
        case[
            "clean_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    current = current_scores(
        case,
        selected,
    )

    before = ndcg(
        current,
        labels,
    )

    utility = []

    for j in candidate_indices:

        after_scores = (
            current.copy()
        )

        after_scores[
            j
        ] = clean[
            j
        ]

        after = ndcg(
            after_scores,
            labels,
        )

        utility.append(
            after - before
        )

    return np.asarray(
        utility,
        dtype=float,
    )


# ============================================================
# Target-independent training state generation
#
# State policies:
#   1. Base-score top prefix
#   2. Random subsets
#
# IMPORTANT:
# State construction does NOT use target identity or utility.
# ============================================================

def generate_states(
    case,
    rng,
):

    n = len(case)

    max_k = max(
        max(
            1,
            int(
                round(
                    b * n
                )
            ),
        )
        for b in BUDGETS
    )

    base = (
        case[
            "base_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    base_order = np.argsort(
        -base,
        kind="stable",
    )

    states = {
        tuple()
    }

    for size in range(
        1,
        max_k,
    ):

        # Base-score prefix.
        states.add(
            tuple(
                sorted(
                    base_order[
                        :size
                    ].tolist()
                )
            )
        )

        # Random states.
        for _ in range(
            N_RANDOM_STATES_PER_SIZE
        ):

            chosen = (
                rng.choice(
                    n,
                    size=size,
                    replace=False,
                )
                .tolist()
            )

            states.add(
                tuple(
                    sorted(
                        chosen
                    )
                )
            )

    return sorted(
        states,
        key=lambda x:
            (
                len(x),
                x,
            )
    )


# ============================================================
# Build supervised datasets
# ============================================================

def build_static_training(
    train,
    base_features,
):

    rows = []
    targets = []

    for _, case in train.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        candidates = list(
            range(
                len(case)
            )
        )

        y = conditional_utilities(
            case,
            [],
            candidates,
        )

        x = case[
            base_features
        ].copy()

        rows.append(x)
        targets.append(y)

    X = pd.concat(
        rows,
        ignore_index=True,
    )

    y = np.concatenate(
        targets
    )

    return X, y


def build_conditional_training(
    train,
    base_features,
):

    rng = np.random.default_rng(
        SEED
    )

    x_parts = []
    y_cond_parts = []
    y_resid_parts = []

    total_states = 0

    case_count = 0

    for _, case in train.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        states = generate_states(
            case,
            rng,
        )

        n = len(case)

        all_candidates = list(
            range(n)
        )

        # ----------------------------------------------------
        # Singleton / intrinsic utility:
        #
        # Δ_i(∅)
        #
        # Compute once per case.
        # ----------------------------------------------------

        singleton_utility = (
            conditional_utilities(
                case,
                [],
                all_candidates,
            )
        )

        for state in states:

            selected = list(
                state
            )

            selected_set = set(
                selected
            )

            remaining = [
                j
                for j in range(n)
                if j
                not in selected_set
            ]

            if not remaining:
                continue

            X = build_action_features(
                case,
                selected,
                remaining,
                base_features,
            )

            # ------------------------------------------------
            # Full conditional marginal:
            #
            # Δ_i(S)
            # ------------------------------------------------

            y_cond = (
                conditional_utilities(
                    case,
                    selected,
                    remaining,
                )
            )

            remaining_idx = np.asarray(
                remaining,
                dtype=int,
            )

            # ------------------------------------------------
            # Ranking-externality residual:
            #
            # r_i(S)
            # =
            # Δ_i(S) - Δ_i(∅)
            #
            # By definition:
            #
            # r_i(∅) = 0
            # ------------------------------------------------

            if len(selected) == 0:

                y_resid = np.zeros_like(
                    y_cond
                )

            else:

                y_resid = (
                    y_cond
                    -
                    singleton_utility[
                        remaining_idx
                    ]
                )

            x_parts.append(
                X
            )

            y_cond_parts.append(
                y_cond
            )

            y_resid_parts.append(
                y_resid
            )

            total_states += 1

        case_count += 1

        if (
            case_count % 500
            == 0
        ):

            print(
                "training-state cases:",
                case_count,
                "states:",
                total_states,
            )

    X = pd.concat(
        x_parts,
        ignore_index=True,
    )

    y_cond = np.concatenate(
        y_cond_parts
    )

    y_resid = np.concatenate(
        y_resid_parts
    )

    return (
        X,
        y_cond,
        y_resid,
        total_states,
    )



# ============================================================
# V3.2: policy-induced / matched state generation
# ============================================================

def rollout_residual_policy_states(
    case,
    static_model,
    residual_model,
    base_features,
):

    n = len(case)

    max_k = max(
        max(
            1,
            int(
                round(
                    b * n
                )
            ),
        )
        for b in BUDGETS
    )

    static_prediction = (
        static_model.predict(
            case[
                base_features
            ]
        )
    )

    selected = []

    remaining = set(
        range(n)
    )

    policy_states = {}

    # --------------------------------------------------------
    # Generate the states actually visited by the current
    # residual policy.
    #
    # For max_k=6, we need states of sizes 1..5 because the
    # state before action 6 contains 5 selected candidates.
    # --------------------------------------------------------

    for _ in range(
        1,
        max_k,
    ):

        candidates = sorted(
            remaining
        )

        if not candidates:
            break

        candidate_idx = np.asarray(
            candidates,
            dtype=int,
        )

        prediction = (
            static_prediction[
                candidate_idx
            ]
            .copy()
        )

        # r_i(empty) = 0.
        # After the first action, use the off-policy residual
        # model to generate the policy-induced trajectory.
        if selected:

            X_residual = (
                build_action_features(
                    case,
                    selected,
                    candidates,
                    base_features,
                )
            )

            prediction = (
                prediction
                +
                residual_model.predict(
                    X_residual
                )
            )

        best_pos = int(
            np.argmax(
                prediction
            )
        )

        chosen = (
            candidates[
                best_pos
            ]
        )

        selected.append(
            chosen
        )

        remaining.remove(
            chosen
        )

        policy_states[
            len(selected)
        ] = tuple(
            sorted(
                selected
            )
        )

    return policy_states


def generate_matched_states(
    case,
    rng,
    policy_states,
):

    n = len(case)

    max_k = max(
        max(
            1,
            int(
                round(
                    b * n
                )
            ),
        )
        for b in BUDGETS
    )

    base = (
        case[
            "base_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    base_order = np.argsort(
        -base,
        kind="stable",
    )

    # --------------------------------------------------------
    # Keep approximately the SAME state count as V3.1:
    #
    # V3.1:
    #   1 base-prefix
    # + 2 random
    #
    # V3.2:
    #   1 base-prefix
    # + 1 policy-induced
    # + random states until the same number of distinct states
    #
    # This prevents "more training rows" from explaining gains.
    # --------------------------------------------------------

    states = [
        tuple()
    ]

    target_per_size = (
        1
        +
        N_RANDOM_STATES_PER_SIZE
    )

    for size in range(
        1,
        max_k,
    ):

        size_states = []

        def add_state(state):

            state = tuple(
                sorted(
                    state
                )
            )

            if state not in size_states:

                size_states.append(
                    state
                )

        # 1. Base-score prefix
        add_state(
            base_order[
                :size
            ].tolist()
        )

        # 2. Policy-induced state
        if size in policy_states:

            add_state(
                policy_states[
                    size
                ]
            )

        # 3. Fill with random distinct states so the number of
        # training states remains matched to V3.1.
        attempts = 0

        while (
            len(size_states)
            <
            target_per_size
        ):

            chosen = (
                rng.choice(
                    n,
                    size=size,
                    replace=False,
                )
                .tolist()
            )

            add_state(
                chosen
            )

            attempts += 1

            if attempts > 10000:
                raise RuntimeError(
                    "Unable to generate enough "
                    "distinct matched states."
                )

        states.extend(
            size_states[
                :target_per_size
            ]
        )

    return states


def build_matched_conditional_training(
    train,
    base_features,
    static_model,
    rollout_residual_model,
):

    rng = np.random.default_rng(
        SEED + 3200
    )

    x_parts = []
    y_resid_parts = []

    total_states = 0

    case_count = 0

    for _, case in train.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        n = len(case)

        all_candidates = list(
            range(n)
        )

        singleton_utility = (
            conditional_utilities(
                case,
                [],
                all_candidates,
            )
        )

        # ----------------------------------------------------
        # Collect policy-induced states using the V3.1
        # residual policy.
        # ----------------------------------------------------

        policy_states = (
            rollout_residual_policy_states(
                case,
                static_model,
                rollout_residual_model,
                base_features,
            )
        )

        # ----------------------------------------------------
        # Construct matched-size training states.
        # ----------------------------------------------------

        states = generate_matched_states(
            case,
            rng,
            policy_states,
        )

        for state in states:

            selected = list(
                state
            )

            selected_set = set(
                selected
            )

            remaining = [
                j
                for j in range(n)
                if j not in selected_set
            ]

            if not remaining:
                continue

            X = build_action_features(
                case,
                selected,
                remaining,
                base_features,
            )

            conditional_y = (
                conditional_utilities(
                    case,
                    selected,
                    remaining,
                )
            )

            remaining_idx = np.asarray(
                remaining,
                dtype=int,
            )

            if len(selected) == 0:

                residual_y = np.zeros_like(
                    conditional_y
                )

            else:

                residual_y = (
                    conditional_y
                    -
                    singleton_utility[
                        remaining_idx
                    ]
                )

            x_parts.append(
                X
            )

            y_resid_parts.append(
                residual_y
            )

            total_states += 1

        case_count += 1

        if (
            case_count % 500
            == 0
        ):

            print(
                "matched-state cases:",
                case_count,
                "states:",
                total_states,
            )

    X = pd.concat(
        x_parts,
        ignore_index=True,
    )

    y_resid = np.concatenate(
        y_resid_parts
    )

    return (
        X,
        y_resid,
        total_states,
    )


# ============================================================
# Policy evaluation
# ============================================================

def final_metrics(
    case,
    selected,
):

    labels = (
        case[
            "is_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    scores = current_scores(
        case,
        selected,
    )

    return ranking_metrics(
        scores,
        labels,
    )


def realized_sequence_utility(
    case,
    selected,
):

    labels = (
        case[
            "is_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    clean = (
        case[
            "clean_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    current = current_scores(
        case,
        [],
    )

    current_ndcg = ndcg(
        current,
        labels,
    )

    marginals = []

    for j in selected:

        new_scores = (
            current.copy()
        )

        new_scores[
            j
        ] = clean[
            j
        ]

        new_ndcg = ndcg(
            new_scores,
            labels,
        )

        marginals.append(
            new_ndcg
            -
            current_ndcg
        )

        current = new_scores
        current_ndcg = new_ndcg

    return np.asarray(
        marginals,
        dtype=float,
    )


def evaluate_static_case(
    case,
    static_model,
    base_features,
    budget,
    abstain,
):

    pred = static_model.predict(
        case[
            base_features
        ]
    )

    n = len(case)

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
            ] > 0.0
        ]

    selected = (
        order[
            :k
        ]
        .tolist()
    )

    return selected


def evaluate_conditional_case(
    case,
    conditional_model,
    base_features,
    budget,
    abstain,
):

    n = len(case)

    k = max(
        1,
        int(
            round(
                budget * n
            )
        ),
    )

    selected = []

    remaining = set(
        range(n)
    )

    for _ in range(k):

        candidates = sorted(
            remaining
        )

        if not candidates:
            break

        X = build_action_features(
            case,
            selected,
            candidates,
            base_features,
        )

        pred = (
            conditional_model.predict(
                X
            )
        )

        best_pos = int(
            np.argmax(
                pred
            )
        )

        best_pred = float(
            pred[
                best_pos
            ]
        )

        if (
            abstain
            and
            best_pred <= 0.0
        ):
            break

        chosen = (
            candidates[
                best_pos
            ]
        )

        selected.append(
            chosen
        )

        remaining.remove(
            chosen
        )

    return selected


def evaluate_residual_case(
    case,
    static_model,
    residual_model,
    base_features,
    budget,
    abstain,
    use_dynamic,
):

    n = len(case)

    k = max(
        1,
        int(
            round(
                budget * n
            )
        ),
    )

    # --------------------------------------------------------
    # Intrinsic singleton utility estimate:
    #
    # û_i(∅)
    #
    # This never changes during the trajectory.
    # --------------------------------------------------------

    static_prediction = (
        static_model.predict(
            case[
                base_features
            ]
        )
    )

    selected = []

    remaining = set(
        range(n)
    )

    for _ in range(k):

        candidates = sorted(
            remaining
        )

        if not candidates:
            break

        candidate_idx = np.asarray(
            candidates,
            dtype=int,
        )

        prediction = (
            static_prediction[
                candidate_idx
            ]
            .copy()
        )

        # ----------------------------------------------------
        # r_i(∅) = 0 exactly.
        #
        # Only after at least one augmentation has actually
        # occurred do we estimate ranking externality.
        # ----------------------------------------------------

        if selected:

            if use_dynamic:

                X_residual = (
                    build_action_features(
                        case,
                        selected,
                        candidates,
                        base_features,
                    )
                )

            else:

                # Same residual labels / same conditional rows
                # during training, but no dynamic state.
                X_residual = (
                    case.iloc[
                        candidate_idx
                    ][
                        base_features
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

            residual_prediction = (
                residual_model.predict(
                    X_residual
                )
            )

            prediction = (
                prediction
                +
                residual_prediction
            )

        best_pos = int(
            np.argmax(
                prediction
            )
        )

        best_pred = float(
            prediction[
                best_pos
            ]
        )

        if (
            abstain
            and
            best_pred <= 0.0
        ):

            break

        chosen = (
            candidates[
                best_pos
            ]
        )

        selected.append(
            chosen
        )

        remaining.remove(
            chosen
        )

    return selected


def evaluate_oracle_case(
    case,
    budget,
    abstain,
):

    n = len(case)

    k = max(
        1,
        int(
            round(
                budget * n
            )
        ),
    )

    selected = []

    remaining = set(
        range(n)
    )

    for _ in range(k):

        candidates = sorted(
            remaining
        )

        if not candidates:
            break

        utility = (
            conditional_utilities(
                case,
                selected,
                candidates,
            )
        )

        best_pos = int(
            np.argmax(
                utility
            )
        )

        best_gain = float(
            utility[
                best_pos
            ]
        )

        if (
            abstain
            and
            best_gain <= EPS
        ):
            break

        chosen = (
            candidates[
                best_pos
            ]
        )

        selected.append(
            chosen
        )

        remaining.remove(
            chosen
        )

    return selected


def evaluate_method(
    val,
    method_name,
    chooser,
    budget,
):

    rows = []

    for case_id, case in val.groupby(
        "case_id",
        sort=False,
    ):

        case = case.reset_index(
            drop=True
        )

        selected = chooser(
            case,
            budget,
        )

        metrics = final_metrics(
            case,
            selected,
        )

        marginal = (
            realized_sequence_utility(
                case,
                selected,
            )
        )

        if len(marginal):

            harm_rate = float(
                np.mean(
                    marginal < -EPS
                )
            )

            help_rate = float(
                np.mean(
                    marginal > EPS
                )
            )

            mean_marginal = float(
                np.mean(
                    marginal
                )
            )

        else:

            harm_rate = 0.0
            help_rate = 0.0
            mean_marginal = 0.0

        rows.append({
            "case_id":
                int(case_id),

            "method":
                method_name,

            "budget":
                float(budget),

            "selected_count":
                int(
                    len(selected)
                ),

            "augment_ratio":
                float(
                    len(selected)
                    /
                    len(case)
                ),

            "ndcg10":
                float(
                    metrics[
                        "ndcg10"
                    ]
                ),

            "hr10":
                float(
                    metrics[
                        "hr10"
                    ]
                ),

            "mrr":
                float(
                    metrics[
                        "mrr"
                    ]
                ),

            "pairwise_win_rate":
                float(
                    metrics[
                        "pairwise_win_rate"
                    ]
                ),

            "realized_harm_rate":
                harm_rate,

            "realized_help_rate":
                help_rate,

            "mean_realized_marginal":
                mean_marginal,
        })

    return pd.DataFrame(
        rows
    )


def summarize_cases(
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

        "realized_harm_rate":
            float(
                cases[
                    "realized_harm_rate"
                ].mean()
            ),

        "realized_help_rate":
            float(
                cases[
                    "realized_help_rate"
                ].mean()
            ),

        "mean_realized_marginal":
            float(
                cases[
                    "mean_realized_marginal"
                ].mean()
            ),
    }


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
        split[
            [
                "case_id",
                "user_id",
                "split",
            ]
        ],
        on=[
            "case_id",
            "user_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if df[
        "split"
    ].isna().any():

        raise RuntimeError(
            "Missing split labels."
        )

    required = {
        "case_id",
        "user_id",
        "base_score",
        "clean_score",
        "is_target",
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

    base_features = [
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
        set(
            base_features
        )
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

    # ========================================================
    # Frozen test split.
    #
    # IMPORTANT:
    # test is NEVER used for fitting, feature design,
    # hyperparameter tuning, or budget selection.
    # ========================================================

    test = (
        df[
            df[
                "split"
            ] == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if test.empty:
        raise RuntimeError(
            "Frozen test split is empty."
        )

    print(
        "============================================================"
    )

    print(
        "SEQUENTIAL MARGINAL V3.2 ON-POLICY"
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
        "base features =",
        len(
            base_features
        )
    )

    print(
        "dynamic features =",
        len(
            DYNAMIC_FEATURES
        )
    )

    print(
        "train cases =",
        train[
            "case_id"
        ].nunique()
    )

    print(
        "val cases =",
        val[
            "case_id"
        ].nunique()
    )

    print(
        "TEST IS NOT USED"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # Static singleton model
    # ========================================================

    print()
    print(
        "===== BUILD STATIC TRAINING DATA ====="
    )

    X_static, y_static = (
        build_static_training(
            train,
            base_features,
        )
    )

    print(
        "static rows =",
        len(
            X_static
        )
    )

    print(
        "static nonzero label rate =",
        float(
            np.mean(
                np.abs(
                    y_static
                )
                > EPS
            )
        )
    )

    static_model = (
        make_xgb(
            SEED
        )
    )

    print(
        "training StaticXGB..."
    )

    static_model.fit(
        X_static,
        y_static,
    )

    # ========================================================
    # Conditional marginal model
    # ========================================================

    print()
    print(
        "===== BUILD CONDITIONAL TRAINING DATA ====="
    )

    (
        X_cond,
        y_cond,
        y_resid,
        n_states,
    ) = build_conditional_training(
        train,
        base_features,
    )

    print(
        "conditional states =",
        n_states
    )

    print(
        "conditional rows =",
        len(
            X_cond
        )
    )

    print(
        "conditional nonzero label rate =",
        float(
            np.mean(
                np.abs(
                    y_cond
                )
                > EPS
            )
        )
    )

    print(
        "conditional positive label rate =",
        float(
            np.mean(
                y_cond
                > EPS
            )
        )
    )

    print(
        "conditional negative label rate =",
        float(
            np.mean(
                y_cond
                < -EPS
            )
        )
    )

    print(
        "residual nonzero label rate =",
        float(
            np.mean(
                np.abs(
                    y_resid
                )
                > EPS
            )
        )
    )

    print(
        "residual positive label rate =",
        float(
            np.mean(
                y_resid
                > EPS
            )
        )
    )

    print(
        "residual negative label rate =",
        float(
            np.mean(
                y_resid
                < -EPS
            )
        )
    )

    print(
        "residual mean abs =",
        float(
            np.mean(
                np.abs(
                    y_resid
                )
            )
        )
    )

    conditional_model = (
        make_xgb(
            SEED + 100
        )
    )

    print(
        "training ConditionalXGB..."
    )

    conditional_model.fit(
        X_cond,
        y_cond,
    )

    # ========================================================
    # Residual externality controls
    # ========================================================

    residual_nodynamic_model = (
        make_xgb(
            SEED + 200
        )
    )

    print(
        "training ResidualXGB_NoDynamic..."
    )

    residual_nodynamic_model.fit(
        X_cond[
            base_features
        ],
        y_resid,
    )

    residual_model = (
        make_xgb(
            SEED + 200
        )
    )

    print(
        "training ResidualXGB..."
    )

    residual_model.fit(
        X_cond,
        y_resid,
    )

    # ========================================================
    # V3.2 matched on-policy residual training
    # ========================================================

    print()
    print(
        "===== BUILD MATCHED ON-POLICY TRAINING DATA ====="
    )

    (
        X_matched,
        y_matched_resid,
        n_matched_states,
    ) = build_matched_conditional_training(
        train,
        base_features,
        static_model,
        residual_model,
    )

    print(
        "matched states =",
        n_matched_states
    )

    print(
        "matched rows =",
        len(
            X_matched
        )
    )

    print(
        "matched residual nonzero rate =",
        float(
            np.mean(
                np.abs(
                    y_matched_resid
                )
                > EPS
            )
        )
    )

    print(
        "matched residual positive rate =",
        float(
            np.mean(
                y_matched_resid
                > EPS
            )
        )
    )

    print(
        "matched residual negative rate =",
        float(
            np.mean(
                y_matched_resid
                < -EPS
            )
        )
    )

    print(
        "matched residual mean abs =",
        float(
            np.mean(
                np.abs(
                    y_matched_resid
                )
            )
        )
    )

    # --------------------------------------------------------
    # Same matched rows + same residual labels.
    #
    # The only difference between these two models is whether
    # dynamic ranking-state features are available.
    # --------------------------------------------------------

    onpolicy_nodynamic_model = (
        make_xgb(
            SEED + 300
        )
    )

    print(
        "training "
        "ResidualXGB_OnPolicy_NoDynamic..."
    )

    onpolicy_nodynamic_model.fit(
        X_matched[
            base_features
        ],
        y_matched_resid,
    )

    onpolicy_residual_model = (
        make_xgb(
            SEED + 300
        )
    )

    print(
        "training "
        "ResidualXGB_OnPolicy..."
    )

    onpolicy_residual_model.fit(
        X_matched,
        y_matched_resid,
    )

    onpolicy_feature_names = (
        base_features
        +
        DYNAMIC_FEATURES
    )

    onpolicy_importance = (
        pd.DataFrame({
            "feature":
                onpolicy_feature_names,

            "importance":
                onpolicy_residual_model
                .named_steps[
                    "model"
                ]
                .feature_importances_,
        })
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    onpolicy_importance.to_csv(
        OUT
        / "onpolicy_residual_feature_importance.csv",
        index=False,
    )

    # Save residual-model feature importance separately.
    residual_feature_names = (
        base_features
        +
        DYNAMIC_FEATURES
    )

    residual_importance = (
        pd.DataFrame({
            "feature":
                residual_feature_names,

            "importance":
                residual_model
                .named_steps[
                    "model"
                ]
                .feature_importances_,
        })
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    residual_importance.to_csv(
        OUT
        / "residual_feature_importance.csv",
        index=False,
    )

    # ========================================================
    # Validation
    # ========================================================

    summary_rows = []
    case_parts = []

    for budget in BUDGETS:

        print()
        print(
            "validation budget =",
            budget
        )

        methods = [
            (
                "StaticXGB",
                lambda case, b:
                    evaluate_static_case(
                        case,
                        static_model,
                        base_features,
                        b,
                        False,
                    ),
            ),
            (
                "ConditionalXGB",
                lambda case, b:
                    evaluate_conditional_case(
                        case,
                        conditional_model,
                        base_features,
                        b,
                        False,
                    ),
            ),
            (
                "ResidualXGB_OffPolicy",
                lambda case, b:
                    evaluate_residual_case(
                        case,
                        static_model,
                        residual_model,
                        base_features,
                        b,
                        False,
                        True,
                    ),
            ),
            (
                "ResidualXGB_OnPolicy_NoDynamic",
                lambda case, b:
                    evaluate_residual_case(
                        case,
                        static_model,
                        onpolicy_nodynamic_model,
                        base_features,
                        b,
                        False,
                        False,
                    ),
            ),
            (
                "ResidualXGB_OnPolicy",
                lambda case, b:
                    evaluate_residual_case(
                        case,
                        static_model,
                        onpolicy_residual_model,
                        base_features,
                        b,
                        False,
                        True,
                    ),
            ),
            (
                "ResidualXGB_OnPolicyAbstain",
                lambda case, b:
                    evaluate_residual_case(
                        case,
                        static_model,
                        onpolicy_residual_model,
                        base_features,
                        b,
                        True,
                        True,
                    ),
            ),
            (
                "OracleGreedyAbstain",
                lambda case, b:
                    evaluate_oracle_case(
                        case,
                        b,
                        True,
                    ),
            ),
        ]

        for method_name, chooser in methods:

            cases = evaluate_method(
                val,
                method_name,
                chooser,
                budget,
            )

            cases[
                "dataset"
            ] = config.DATASET_NAME

            cases[
                "model"
            ] = config.MODEL_NAME

            cases[
                "seed"
            ] = SEED

            case_parts.append(
                cases
            )

            s = summarize_cases(
                cases
            )

            summary_rows.append({
                "dataset":
                    config.DATASET_NAME,

                "model":
                    config.MODEL_NAME,

                "seed":
                    SEED,

                "method":
                    method_name,

                "budget":
                    budget,

                **s,
            })

    summary = pd.DataFrame(
        summary_rows
    )

    case_metrics = pd.concat(
        case_parts,
        ignore_index=True,
    )

    summary.to_csv(
        OUT
        / "validation_budget_curve.csv",
        index=False,
    )

    case_metrics.to_csv(
        OUT
        / "validation_case_metrics.csv",
        index=False,
    )

    selected_rows = []

    for method, g in summary.groupby(
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

    # ========================================================
    # Feature importance for conditional model
    # ========================================================

    feature_names = (
        base_features
        +
        DYNAMIC_FEATURES
    )

    importance = pd.DataFrame({
        "feature":
            feature_names,

        "importance":
            conditional_model
            .named_steps[
                "model"
            ]
            .feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        OUT
        / "conditional_feature_importance.csv",
        index=False,
    )

    manifest = {
        "dataset":
            config.DATASET_NAME,

        "model":
            config.MODEL_NAME,

        "seed":
            SEED,

        "budgets":
            BUDGETS,

        "base_features":
            base_features,

        "dynamic_features":
            DYNAMIC_FEATURES,

        "random_states_per_size":
            N_RANDOM_STATES_PER_SIZE,

        "train_state_generation": [
            "base_score_top_prefix",
            "random_subset",
        ],

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
                "realized_harm_rate",
                "realized_help_rate",
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

    # ========================================================
    # FROZEN TEST EVALUATION
    #
    # CRITICAL:
    # - models were fitted on TRAIN only
    # - operating budget comes from VALIDATION only
    # - TEST is evaluated exactly at that locked budget
    # - no test-budget search is performed
    # ========================================================

    print()
    print(
        "===== FROZEN TEST EVALUATION ====="
    )

    print(
        "test cases =",
        test[
            "case_id"
        ].nunique()
    )

    validation_selected_path = (
        OUT
        / "validation_selected.csv"
    )

    if not validation_selected_path.exists():
        raise FileNotFoundError(
            validation_selected_path
        )

    locked = pd.read_csv(
        validation_selected_path
    )

    # --------------------------------------------------------
    # Audit copy: records exactly which validation-selected
    # operating points were used for test.
    # --------------------------------------------------------

    locked.to_csv(
        OUT
        / "test_selection_lock.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Frozen chooser definitions.
    # No test-dependent threshold/budget/model selection.
    # --------------------------------------------------------

    frozen_test_choosers = {

        "StaticXGB":
            lambda case, b:
                evaluate_static_case(
                    case,
                    static_model,
                    base_features,
                    b,
                    False,
                ),

        "ConditionalXGB":
            lambda case, b:
                evaluate_conditional_case(
                    case,
                    conditional_model,
                    base_features,
                    b,
                    False,
                ),

        "ResidualXGB_OffPolicy":
            lambda case, b:
                evaluate_residual_case(
                    case,
                    static_model,
                    residual_model,
                    base_features,
                    b,
                    False,
                    True,
                ),

        "ResidualXGB_OnPolicy_NoDynamic":
            lambda case, b:
                evaluate_residual_case(
                    case,
                    static_model,
                    onpolicy_nodynamic_model,
                    base_features,
                    b,
                    False,
                    False,
                ),

        "ResidualXGB_OnPolicy":
            lambda case, b:
                evaluate_residual_case(
                    case,
                    static_model,
                    onpolicy_residual_model,
                    base_features,
                    b,
                    False,
                    True,
                ),

        "ResidualXGB_OnPolicyAbstain":
            lambda case, b:
                evaluate_residual_case(
                    case,
                    static_model,
                    onpolicy_residual_model,
                    base_features,
                    b,
                    True,
                    True,
                ),

        "OracleGreedyAbstain":
            lambda case, b:
                evaluate_oracle_case(
                    case,
                    b,
                    True,
                ),
    }


    test_case_parts = []
    test_summary_rows = []

    for _, locked_row in locked.iterrows():

        method_name = str(
            locked_row[
                "method"
            ]
        )

        budget = float(
            locked_row[
                "budget"
            ]
        )

        if (
            method_name
            not in
            frozen_test_choosers
        ):
            raise RuntimeError(
                "Unknown locked method: "
                f"{method_name}"
            )

        print(
            "frozen test:",
            method_name,
            "budget =",
            budget,
        )

        cases = evaluate_method(
            test,
            method_name,
            frozen_test_choosers[
                method_name
            ],
            budget,
        )

        cases.insert(
            0,
            "dataset",
            config.DATASET_NAME,
        )

        cases.insert(
            1,
            "model",
            config.MODEL_NAME,
        )

        cases.insert(
            2,
            "seed",
            SEED,
        )

        test_case_parts.append(
            cases
        )

        summary = summarize_cases(
            cases
        )

        test_summary_rows.append({
            "dataset":
                config.DATASET_NAME,

            "model":
                config.MODEL_NAME,

            "seed":
                SEED,

            "method":
                method_name,

            "budget":
                budget,

            "validation_ndcg10":
                float(
                    locked_row[
                        "ndcg10"
                    ]
                ),

            **summary,
        })


    test_cases = pd.concat(
        test_case_parts,
        ignore_index=True,
    )

    test_summary = pd.DataFrame(
        test_summary_rows
    )

    test_cases.to_csv(
        OUT
        / "test_selected_cases.csv",
        index=False,
    )

    test_summary.to_csv(
        OUT
        / "test_selected_summary.csv",
        index=False,
    )


    print()
    print(
        "===== FROZEN TEST SELECTED ====="
    )

    print(
        test_summary
        .sort_values(
            "ndcg10",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        "===== TOP CONDITIONAL FEATURES ====="
    )

    print(
        importance.head(
            20
        ).to_string(
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
