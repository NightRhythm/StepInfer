"""Qwen3 核心模型结构单元测试，涵盖 MLP、Attention、DecoderLayer、Model 及 ForCausalLM。"""

import pytest
import torch

from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)


@pytest.fixture
def tiny_qwen3_config():
    """构造微型 Qwen3 配置，用于快速单元测试。"""
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        bos_token_id=0,
        eos_token_id=0,
    )


def test_qwen3_mlp_forward_shape_and_dtype(tiny_qwen3_config):
    """验证 Qwen3MLP 升降维与输出形状保持。"""
    mlp = Qwen3MLP(tiny_qwen3_config)
    x = torch.randn(2, 4, tiny_qwen3_config.hidden_size, dtype=torch.float32)

    out = mlp(x)
    assert out.shape == (2, 4, tiny_qwen3_config.hidden_size)
    assert out.dtype == torch.float32


def test_qwen3_attention_forward_shape_and_flow(tiny_qwen3_config):
    """验证 Qwen3Attention 包含 QK-Norm、RoPE 与因果注意力的完整输出形状。"""
    attn = Qwen3Attention(tiny_qwen3_config)
    batch_size, seq_len = 2, 5
    hidden_states = torch.randn(batch_size, seq_len, tiny_qwen3_config.hidden_size)
    positions = torch.arange(seq_len)

    out = attn(positions, hidden_states)
    assert out.shape == (batch_size, seq_len, tiny_qwen3_config.hidden_size)

    # 非法 2D 输入验证
    with pytest.raises(ValueError, match="must be a 3D tensor"):
        attn(positions, hidden_states[0])


def test_qwen3_decoder_layer_residual_flow(tiny_qwen3_config):
    """验证 Qwen3DecoderLayer 的残差流水线 (包含初始残差与跨层残差累加)。"""
    layer = Qwen3DecoderLayer(tiny_qwen3_config)
    batch_size, seq_len = 2, 4
    hidden_states = torch.randn(batch_size, seq_len, tiny_qwen3_config.hidden_size)
    positions = torch.arange(seq_len)

    # 1. 初始层 (residual 为 None)
    out, updated_residual = layer(positions, hidden_states, residual=None)
    assert out.shape == (batch_size, seq_len, tiny_qwen3_config.hidden_size)
    assert updated_residual.shape == (batch_size, seq_len, tiny_qwen3_config.hidden_size)

    # 2. 后续层 (传入上一层的 updated_residual)
    out2, updated_residual2 = layer(positions, out, residual=updated_residual)
    assert out2.shape == (batch_size, seq_len, tiny_qwen3_config.hidden_size)
    assert updated_residual2.shape == (batch_size, seq_len, tiny_qwen3_config.hidden_size)


def test_qwen3_model_forward_shape(tiny_qwen3_config):
    """验证 Qwen3Model 主干模型前向计算形状。"""
    model = Qwen3Model(tiny_qwen3_config)
    input_ids = torch.randint(0, tiny_qwen3_config.vocab_size, (2, 6))

    hidden_states = model(input_ids)
    assert hidden_states.shape == (2, 6, tiny_qwen3_config.hidden_size)

    # 非法 1D input_ids 验证
    with pytest.raises(ValueError, match="input_ids must be a 2D tensor"):
        model(input_ids[0])


def test_qwen3_for_causal_lm_logits_and_weight_tying(tiny_qwen3_config):
    """验证 Qwen3ForCausalLM 输出 Logits 形状与权重绑定 (Weight Tying)。"""
    # 1. 启用权重绑定
    tiny_qwen3_config.tie_word_embeddings = True
    model = Qwen3ForCausalLM(tiny_qwen3_config)
    assert model.lm_head.weight is model.model.embed_tokens.weight

    input_ids = torch.randint(0, tiny_qwen3_config.vocab_size, (1, 8))
    logits = model(input_ids)
    assert logits.shape == (1, 8, tiny_qwen3_config.vocab_size)

    # 2. 未启用权重绑定
    tiny_qwen3_config.tie_word_embeddings = False
    model_no_tie = Qwen3ForCausalLM(tiny_qwen3_config)
    assert model_no_tie.lm_head.weight is not model_no_tie.model.embed_tokens.weight
