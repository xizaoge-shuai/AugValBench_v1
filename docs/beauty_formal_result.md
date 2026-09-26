# Beauty Formal Experiment

## Protocol

- Dataset: Amazon All_Beauty
- Backbone: Qwen2.5-7B-Instruct
- Filtering: 3-core user/item
- Candidate protocol: popularity-matched
- Candidates per request: 20
- Train / Validation / Test:
  - Train: 585 requests
  - Validation: 195 requests
  - Test: 195 requests

## Frozen Method

DirectNDCG:
- RandomForestRegressor
- n_estimators = 600
- max_depth = 12
- min_samples_leaf = 3
- pre-augmentation features only
- validation-selected budget = 10%

## Held-out Test

DirectNDCG @ 10%:
- NDCG@10 = 0.385671
- HR@10 = 0.707692
- MRR = 0.308007
- Pairwise Win Rate = 0.651012

NoAugment:
- NDCG@10 = 0.363983

UniformClean:
- NDCG@10 = 0.363340

Tail @ validation-selected 10%:
- NDCG@10 = 0.369858

BenefitRF_Margin @ validation-selected 10%:
- NDCG@10 = 0.380281

Important:
BenefitRF_NDCG has stronger NDCG at some fixed budgets on the held-out
test, so current evidence supports utility-aware selective augmentation,
but does not establish that direct regression dominates classification
at every fixed budget.

Do not tune the method further on Beauty test.
