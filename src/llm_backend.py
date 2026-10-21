from typing import Sequence, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src import config


class BinaryLLMScorer:

    def __init__(self):

        print("Loading:", config.MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        quant = None

        if config.LOAD_IN_4BIT:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            device_map="auto",
            dtype=torch.float16,
            quantization_config=quant,
            trust_remote_code=not str(config.MODEL_NAME).startswith(
                "microsoft/Phi-3.5-"
            ),
        )

        self.model.eval()

        # Qwen: 单 token
        # Phi/Mistral: 自动支持多 token 标签
        self.tok0 = self._label_token_ids(["0", " 0"])
        self.tok1 = self._label_token_ids(["1", " 1"])
        self.tok2 = self._label_token_ids(["2", " 2"])

        if len({self.tok0, self.tok1, self.tok2}) != 3:
            raise RuntimeError(
                "Label token sequences are not distinct"
            )

        print(
            "label token sequences:",
            self.tok0,
            self.tok1,
            self.tok2,
        )

    def _label_token_ids(self, variants):

        options = []

        for label in variants:

            ids = tuple(
                self.tokenizer.encode(
                    label,
                    add_special_tokens=False,
                )
            )

            # 保持原来的 Qwen 单 token 优先规则
            if len(ids) == 1:
                return ids

            if ids:
                options.append(ids)

        if not options:
            raise RuntimeError(
                f"Cannot tokenize labels: {variants}"
            )

        # 多 token 时优先选择最短编码
        return min(options, key=len)

    def chat_prompt(self, user_text: str) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful recommender-system evaluator. "
                    "Follow the requested output format exactly."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def _last_logits(
        self,
        prompts: Sequence[str],
        prefix_ids=(),
    ) -> torch.Tensor:

        # 预留标签前缀占用的 context 长度
        n_prefix = len(prefix_ids)

        max_length = (
            config.MAX_PROMPT_TOKENS - n_prefix
        )

        if max_length < 1:
            raise ValueError(
                "MAX_PROMPT_TOKENS too small"
            )

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # 逐 token 延长 prompt，计算标签续写概率
        if prefix_ids:

            extra = torch.tensor(
                prefix_ids,
                dtype=enc["input_ids"].dtype,
            )

            extra = (
                extra.unsqueeze(0)
                .expand(
                    enc["input_ids"].shape[0],
                    -1,
                )
            )

            enc["input_ids"] = torch.cat(
                [
                    enc["input_ids"],
                    extra,
                ],
                dim=1,
            )

            if "attention_mask" in enc:

                ones = torch.ones_like(
                    extra,
                    dtype=enc["attention_mask"].dtype,
                )

                enc["attention_mask"] = torch.cat(
                    [
                        enc["attention_mask"],
                        ones,
                    ],
                    dim=1,
                )

            if "token_type_ids" in enc:

                last = enc["token_type_ids"][:, -1:]

                enc["token_type_ids"] = torch.cat(
                    [
                        enc["token_type_ids"],
                        last.expand(-1, n_prefix),
                    ],
                    dim=1,
                )

        device = next(
            self.model.parameters()
        ).device

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        out = self.model(**enc)

        return out.logits[:, -1, :].float()

    @torch.inference_mode()
    def _pair_logprob_difference(
        self,
        batch,
        left,
        right,
    ):
        """
        返回：
            log P(right | prompt)
            -
            log P(left | prompt)

        对公共标签前缀进行抵消。

        对 Qwen 的单 token 标签，等价于原始 logit 差。
        对 Phi 的公共前缀双 token 标签，只需一次模型前向。
        """

        common = 0

        while (
            common < min(len(left), len(right))
            and left[common] == right[common]
        ):
            common += 1

        if (
            common == len(left)
            or common == len(right)
        ):
            raise RuntimeError(
                "One label sequence prefixes another"
            )

        # Qwen 单 token，或者 Phi 共享一个前缀 token
        if (
            len(left) == common + 1
            and len(right) == common + 1
        ):

            logits = self._last_logits(
                batch,
                left[:common],
            )

            return (
                logits[:, right[-1]]
                -
                logits[:, left[-1]]
            )

        # 通用多 token 方案
        def remaining_logp(ids):

            total = None

            for i in range(common, len(ids)):

                logits = self._last_logits(
                    batch,
                    ids[:i],
                )

                lp = torch.log_softmax(
                    logits,
                    dim=-1,
                )[:, ids[i]]

                total = (
                    lp
                    if total is None
                    else total + lp
                )

            return total

        return (
            remaining_logp(right)
            -
            remaining_logp(left)
        )

    def prob_one_vs_zero(
        self,
        prompts: Sequence[str],
    ) -> List[float]:

        results = []

        for st in range(
            0,
            len(prompts),
            config.BATCH_SIZE,
        ):

            batch = prompts[
                st:st + config.BATCH_SIZE
            ]

            diff = self._pair_logprob_difference(
                batch,
                self.tok0,
                self.tok1,
            )

            p = torch.sigmoid(diff)

            results.extend(
                p.detach().cpu().tolist()
            )

        return results

    def option1_minus_option2_margin(
        self,
        prompts: Sequence[str],
    ) -> List[float]:
        """
        返回：
            log P("1"|prompt)
            -
            log P("2"|prompt)
        """

        results = []

        for st in range(
            0,
            len(prompts),
            config.BATCH_SIZE,
        ):

            batch = prompts[
                st:st + config.BATCH_SIZE
            ]

            margin = self._pair_logprob_difference(
                batch,
                self.tok2,
                self.tok1,
            )

            results.extend(
                margin.detach().cpu().tolist()
            )

        return results
