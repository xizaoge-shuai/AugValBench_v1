# AugValBench preliminary

第一阶段只验证一个问题：

> **Knowledge Gap Does Not Imply Knowledge Gain**

即：

> LLM“不知道”一个 item，不代表补充 item 信息后推荐一定更好。

## 1. 环境

环境名用最简单的：

```bash
conda create -n rec python=3.11 -y
conda activate rec
```

安装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装其余依赖：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 第一次先开 DEBUG

修改 `config.py`：

```python
DEBUG = True
DEBUG_USERS = 20
BATCH_SIZE = 4
```

跑通后再改成：

```python
DEBUG = False
MAX_EVAL_USERS = 200
```

## 3. 运行顺序

```bash
python prepare_amazon.py
python build_probes.py
python probe_knowledge.py
python run_rerank.py
python run_baselines.py
python analyze_gain.py
```

全部确认无误后可以：

```bash
python run_all.py
```

## 4. 数据

默认：

```text
Amazon Reviews'23 / All_Beauty
```

采用 leave-last-out：

```text
历史交互 -> user history
最后一次交互 -> target
```

每个测试请求：

```text
1 positive + 19 negatives
```

## 5. 模型

默认：

```text
Qwen/Qwen2.5-3B-Instruct
4-bit
```

显存充足可在 `config.py` 改成：

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

## 6. AugValBench 条件

```text
base
clean
redundant
partial
irrelevant
conflict
```

暂时不做 stale，因为 Amazon 静态 metadata 没有 item 历史版本，硬造不严谨。

## 7. Knowledge probe

`build_probes.py` 构造：

```text
anchor item
+ 真实 collaborative neighbor
+ 随机 decoy
```

然后 `probe_knowledge.py` 让 LLM 做二选一。

输出：

```text
results/item_knowledge.csv
```

其中：

```text
knowledge_score 越高
= LLM 越容易识别真实协同邻居
```

这只是 KnowSA comparative knowledge probing 的“思想级复现”，不是官方代码。

## 8. Ranking

`run_rerank.py` 不让 LLM 生成长答案，而直接比较下一 token：

```text
0 = unlikely
1 = likely
```

定义：

```text
score = P(1) / (P(0) + P(1))
```

再对 candidate 排序。

## 9. 思想级 baseline

`run_baselines.py` 包含：

```text
NoAugment
UniformClean
Random50
TailSelective
LowKnowledgeSelective
OracleUpperBound
```

其中：

```text
LowKnowledgeSelective
```

代表：

```text
knowledge poor -> augment
```

这是 KnowSA 的核心选择思想。

`OracleUpperBound` 使用 ground truth，只用于判断 method space，不能当正式方法。

## 10. 最重要的三个判断门

### Gate A

看：

```text
Clean harmful augmentation rate
Clean helpful augmentation rate
```

如果 clean augmentation 对一部分请求有帮助、另一部分请求反而有害，说明 augmentation utility 是异质的。

### Gate B

看：

```text
Spearman(knowledge_score, delta_ndcg10)
```

如果相关性不强，尤其存在：

```text
knowledge low 但 gain <= 0
knowledge high 但 gain > 0
```

就支持：

```text
Knowledge Gap != Augmentation Utility
```

### Gate C

看：

```text
results/baseline_summary.csv
```

比较：

```text
LowKnowledgeSelective
vs
OracleUpperBound
```

如果 Oracle 明显高，说明真正的问题是：

```text
“该不该增强”没有选对
```

这时第二阶段才值得训练：

```text
Benefit Predictor
```

直接预测：

```text
DeltaU_hat(x, a)
```
