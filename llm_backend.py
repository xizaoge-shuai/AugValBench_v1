from typing import Sequence, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import config


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
            torch_dtype=torch.float16,
            quantization_config=quant,
            trust_remote_code=True,
        )
        self.model.eval()

        self.tok0 = self._single_token_id(["0", " 0"])
        self.tok1 = self._single_token_id(["1", " 1"])
        self.tok2 = self._single_token_id(["2", " 2"])
        print("label ids:", self.tok0, self.tok1, self.tok2)

    def _single_token_id(self, variants):
        for text in variants:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise RuntimeError(f"找不到单 token 标签：{variants}")

    def chat_prompt(self, user_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful recommender-system evaluator. "
                    "Follow the requested output format exactly."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def _last_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_PROMPT_TOKENS,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc)
        return out.logits[:, -1, :].float()

    def prob_one_vs_zero(self, prompts: Sequence[str]) -> List[float]:
        results = []
        for st in range(0, len(prompts), config.BATCH_SIZE):
            batch = prompts[st: st + config.BATCH_SIZE]
            logits = self._last_logits(batch)
            pair = logits[:, [self.tok0, self.tok1]]
            p = torch.softmax(pair, dim=-1)[:, 1]
            results.extend(p.detach().cpu().tolist())
        return results

    def option1_minus_option2_margin(self, prompts: Sequence[str]) -> List[float]:
        """Return logit('1') - logit('2') for each prompt."""
        results = []
        for st in range(0, len(prompts), config.BATCH_SIZE):
            batch = prompts[st: st + config.BATCH_SIZE]
            logits = self._last_logits(batch)
            margin = logits[:, self.tok1] - logits[:, self.tok2]
            results.extend(margin.detach().cpu().tolist())
        return results
