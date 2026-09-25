# AugValBench v2 patch

这次修改针对 DEBUG 暴露出的两个问题：

1. **97.6% 的 Option-2 position bias**
2. **弱协同正邻居 / 偶然共现噪声**

同时增加更连续的 augmentation utility 指标。

## 替换文件

```text
config.py
build_probes.py
llm_backend.py
probe_knowledge.py
run_rerank.py
analyze_gain.py
```

## v2 position debias

同一道 probe 跑两个顺序：

```text
Forward:
Option1 = positive
Option2 = negative

Swapped:
Option1 = negative
Option2 = positive
```

令：

```text
m_f = logit(1)-logit(2) on Forward
m_s = logit(1)-logit(2) on Swapped
```

则：

```text
semantic_margin = (m_f - m_s)/2
position_bias   = (m_f + m_s)/2
knowledge_score = sigmoid(semantic_margin)
```

固定的位置偏置被抵消。

## v2 positive / negative

正邻居：

```text
co_count >= 2
优先高 co_count + 高 association strength
```

不再 fallback 到 co_count=1。

负邻居：

```text
与 anchor 从未共现
且 popularity 尽量接近
```

## 新 utility 指标

除 NDCG 外增加：

```text
pairwise_win_rate
pairwise_margin
```

`pairwise_win_rate` 是 target score 胜过 negative 的比例，比单请求 NDCG 更连续。

## 重跑

不用重新下载 Amazon，也不用重新跑 `prepare_amazon.py`。

保持：

```python
DEBUG = True
DEBUG_USERS = 20
```

运行：

```bash
python build_probes.py
python probe_knowledge.py
python run_rerank.py
python run_baselines.py
python analyze_gain.py
```

确认 v2 probe 正常后，再扩大到 200 users。
