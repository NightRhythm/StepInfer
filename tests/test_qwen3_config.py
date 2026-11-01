"""Qwen3Config 单元测试，涵盖默认值、维度回退、边界约束与文件解析。"""

import json
from pathlib import Path
import pytest

from stepinfer.config.qwen3_config import Qwen3Config


def test_default_config_initialization():
    """验证默认配置参数与 Qwen3-0.6B-Base 对齐。"""
    config = Qwen3Config()
    assert config.vocab_size == 151936
    assert config.hidden_size == 1024
    assert config.intermediate_size == 3072
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 1000000.0
    assert config.tie_word_embeddings is True


def test_head_dim_auto_fallback_when_zero_or_none():
    """验证当 head_dim 未设置或 <= 0 时，自动回退计算为 hidden_size // num_attention_heads。"""
    config = Qwen3Config(hidden_size=512, num_attention_heads=8, head_dim=0)
    assert config.head_dim == 64

    # 无法整除时应抛出异常
    with pytest.raises(ValueError, match="必须能被"):
        Qwen3Config(hidden_size=500, num_attention_heads=8, head_dim=0)


@pytest.mark.parametrize(
    "field,invalid_val",
    [
        ("vocab_size", 0),
        ("hidden_size", -10),
        ("intermediate_size", 0),
        ("num_hidden_layers", 0),
        ("num_attention_heads", -1),
        ("num_key_value_heads", 0),
        ("rms_norm_eps", -1e-5),
        ("rope_theta", 0),
        ("max_position_embeddings", -100),
    ],
)
def test_positive_field_validation(field, invalid_val):
    """验证关键维度与超参数小于等于 0 时抛出异常。"""
    kwargs = {field: invalid_val}
    with pytest.raises(ValueError, match="must be > 0"):
        Qwen3Config(**kwargs)


def test_token_id_range_validation():
    """验证 bos_token_id 和 eos_token_id 越界检查。"""
    # 负数 ID
    with pytest.raises(ValueError, match="bos_token_id must be >= 0"):
        Qwen3Config(bos_token_id=-1)

    with pytest.raises(ValueError, match="eos_token_id must be >= 0"):
        Qwen3Config(eos_token_id=-1)

    # 超过词表大小 (当缩小 vocab_size 时，需同时指定合法的另一个 token_id 以便单独触发目标检查)
    with pytest.raises(ValueError, match="bos_token_id must be >= 0 and < 100"):
        Qwen3Config(vocab_size=100, bos_token_id=100, eos_token_id=0)

    with pytest.raises(ValueError, match="eos_token_id must be >= 0 and < 100"):
        Qwen3Config(vocab_size=100, bos_token_id=0, eos_token_id=105)


def test_gqa_divisibility_validation():
    """验证 GQA 头数整除性约束。"""
    # 正常整除 (16 可以被 8 整除)
    config = Qwen3Config(num_attention_heads=16, num_key_value_heads=8)
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8

    # 无法整除时抛出 ValueError
    with pytest.raises(ValueError, match="GQA 配置错误"):
        Qwen3Config(num_attention_heads=16, num_key_value_heads=5)


def test_from_dict_filters_unknown_parameters():
    """验证从包含未知额外字段的字典解析时自动过滤。"""
    raw_dict = {
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        # 冗余字段（如 HuggingFace 特有字段）
        "architectures": ["Qwen3ForCausalLM"],
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "random_extra_param": 12345,
    }
    config = Qwen3Config.from_dict(raw_dict)
    assert config.vocab_size == 32000
    assert config.hidden_size == 2048
    assert config.num_key_value_heads == 4
    assert not hasattr(config, "random_extra_param")


def test_from_json_file_loads_real_model_config():
    """验证能够成功解析真实的本地 Qwen3-0.6B config.json。"""
    config_path = "assets/models/Qwen3-0.6B-Base/config.json"
    config = Qwen3Config.from_json_file(config_path)

    assert config.vocab_size == 151936
    assert config.hidden_size == 1024
    assert config.intermediate_size == 3072
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.rope_theta == 1000000.0


def test_from_json_file_missing_file_raises_not_found(tmp_path: Path):
    """验证文件不存在时抛出 FileNotFoundError。"""
    non_existent = tmp_path / "non_existent_config.json"
    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        Qwen3Config.from_json_file(non_existent)
