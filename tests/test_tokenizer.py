"""分词器 (HFTokenizer) 单元测试。"""

import pytest
import torch

from stepinfer.config.model_config import load_model_config
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return HFTokenizer(load_model_config())


def test_encode_decode_round_trip(tokenizer):
    """验证单序列在 Tensor 与 List 两种模式下的编解码往返一致性。"""
    text = "你好，stepinfer"

    # 1. 默认 PyTorch Tensor 输出 [1, seq_len]
    tensor_ids = tokenizer.encode(text)
    assert isinstance(tensor_ids, torch.Tensor)
    assert tensor_ids.ndim == 2
    assert tokenizer.decode(tensor_ids) == text

    # 2. 原生 Python List 输出
    list_ids = tokenizer.encode(text, return_tensors=None)
    assert isinstance(list_ids, list)
    assert tokenizer.decode(list_ids) == text


def test_batch_encode_decode_round_trip(tokenizer):
    """验证多序列批处理编解码一致性与 Attention Mask 形状。"""
    prompts = ["first", "second sentence"]
    input_ids, attention_mask = tokenizer.encode(prompts)

    assert input_ids.shape[0] == 2
    assert attention_mask.shape == input_ids.shape
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    assert decoded == prompts


def test_special_tokens_properties(tokenizer):
    """验证关键特殊 Token ID 属性可用。"""
    assert isinstance(tokenizer.eos_token_id, int)
    assert tokenizer.pad_token_id is not None
