# Current Method Prototype

## Development benchmark

- Dataset: Amazon All_Beauty
- Candidate protocol: popularity-matched negatives
- Evaluation requests: 200
- Backbone: Qwen2.5-7B-Instruct
- Candidates/request: 20
- Conditions used for counterfactual construction:
  - base
  - clean

## Main method prototype

DirectNDCG candidate-level utility prediction.

For each candidate i:

DeltaNDCG_i =
NDCG(candidate i augmented, others base)
-
NDCG(all candidates base)

Only pre-augmentation features are allowed.

The regressor predicts:

    DeltaNDCG_hat(u, i)

At augmentation budget B:

    select Top-B candidates
    according to predicted DeltaNDCG.

## Current operating budgets

Primary:
- 10%
- 15%
- 20%

Secondary:
- 5%
- 30%

## Development baselines

- NoAugment
- UniformClean
- Random
- Tail
- LowKnowledge
- BenefitRF
- UtilityMarginRF
- HurdlePWR
- HurdleNDCG
- OraclePWR
- OracleNDCG

## Important rule

The current 200-request benchmark is a DEVELOPMENT set.

Do not report it as the final held-out test set.

All final method choices and hyperparameters must be frozen
before the full Beauty Train/Validation/Test experiment.
