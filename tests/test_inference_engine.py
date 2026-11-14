"""推理引擎 (InferenceEngine) 快速 Mock 单元测试。"""

import pytest
import torch
from torch import nn

from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.tokenizer.base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    @property
    def eos_token_id(self):
        return 2

    def encode(self, text, **kwargs):
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)

    def decode(self, token_ids, **kwargs):
        if token_ids.ndim == 2:
            token_ids = token_ids[0]
        return "".join(chr(token_id) for token_id in token_ids.tolist())


class ConstantLogitsModel(nn.Module):
    def __init__(self, next_token_id: int):
        super().__init__()
        self.next_token_id = next_token_id

    def forward(self, input_ids):
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], 128, device=input_ids.device
        )
        logits[..., self.next_token_id] = 1.0
        return logits


def test_engine_generates_text_and_lifecycle():
    """验证 InferenceEngine 驱动模型自回归生成、自动设为 eval 模式及返回完整结果字典。"""
    model = ConstantLogitsModel(next_token_id=33)  # ord('!') == 33
    model.train()

    engine = InferenceEngine(model=model, tokenizer=CharacterTokenizer())
    assert engine.model.training is False

    result = engine.generate("hi", max_new_tokens=2)
    assert result == {
        "prompt": "hi",
        "generated_text": "hi!!",
        "sampling_method": "greedy",
        "token_count": 2,
        "stop_reason": "Reached max_new_tokens limit of 2.",
    }


def test_engine_input_boundaries():
    """验证非法输入边界（空 Prompt、非正 token 上限）的防御拦截。"""
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
    )

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        engine.generate("   ")

    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        engine.generate("hi", max_new_tokens=0)
