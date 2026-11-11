"""批量推理 (Batch Inference) 与左填充掩码 (Left-Padding Mask) 单元测试。"""

import pytest
import torch
from torch import nn

from stepinfer.config.model_config import load_model_config
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.generator import generate_tokens
from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.layers.attention import CausalAttention
from stepinfer.models.qwen3 import Qwen3ForCausalLM
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return HFTokenizer(load_model_config())


def test_batch_encode_left_padding_structure(tokenizer):
    """验证 encode 接收列表时默认左填充结构与 attention_mask 正确性。"""
    prompts = ["Hi", "The quick brown fox jumps over the lazy dog."]
    input_ids, attention_mask = tokenizer.encode(prompts, padding_side="left")

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.shape == attention_mask.shape
    assert input_ids.ndim == 2
    assert input_ids.shape[0] == 2

    # 第一个句子短，左侧必须包含 PAD，mask 左侧为 0
    pad_id = tokenizer.pad_token_id
    assert input_ids[0, 0].item() == pad_id
    assert attention_mask[0, 0].item() == 0

    # 第二个句子最长，最左侧不能是 PAD，mask 最左侧为 1
    assert attention_mask[1, 0].item() == 1

    # 两条序列的末尾均为有效 token，mask 均为 1
    assert attention_mask[0, -1].item() == 1
    assert attention_mask[1, -1].item() == 1


def test_batch_decode(tokenizer):
    """验证 decode 解码批量 token ID。"""
    prompts = ["Hello", "World"]
    input_ids, _ = tokenizer.encode(prompts)
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)

    assert len(decoded) == 2
    assert decoded[0] == "Hello"
    assert decoded[1] == "World"


def test_attention_padding_mask_prevents_nan():
    """验证全 PAD 行为使用 finfo.min 不产生 NaN。"""
    attn = CausalAttention(num_heads=4, head_dim=16)
    batch_size, seq_len = 2, 4
    q = torch.randn(batch_size, seq_len, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, 4, 16, dtype=torch.bfloat16)

    # 序列 0 前两项为 PAD
    padding_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long)

    out = attn(q, k, v, padding_mask=padding_mask)

    assert out.shape == q.shape
    assert not torch.isnan(out).any(), "Attention output contains NaN!"


def test_batch_generator_per_sequence_eos_tracking():
    """验证批量生成时各序列独立停止与陪跑冻结机制。"""
    class MockBatchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.step = 0

        def forward(self, input_ids):
            batch_size = input_ids.shape[0]
            vocab_size = 30
            logits = torch.zeros(batch_size, input_ids.shape[1], vocab_size)
            # 序列 0 在首步吐出 EOS=2；序列 1 持续吐出 token 5
            logits[0, -1, 2] = 10.0
            logits[1, -1, 5] = 10.0
            return logits

    model = MockBatchModel()
    input_ids = torch.tensor([[10, 11], [12, 13]])
    padding_mask = torch.tensor([[1, 1], [1, 1]])

    gen_ids, token_counts, stop_reasons = generate_tokens(
        model=model,
        input_ids=input_ids,
        padding_mask=padding_mask,
        device="cpu",
        max_new_tokens=4,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )

    # 序列 0 应该在第 1 步就因 EOS 停止
    assert token_counts[0] == 1
    assert stop_reasons[0] == "Generated EOS token."

    # 序列 1 应该跑满 4 步
    assert token_counts[1] == 4
    assert stop_reasons[1] == "Reached max_new_tokens limit of 4."

    # 序列 0 后续陪跑步骤应该被冻结为 pad_token_id=0
    # 总长度为 2 (prompt) + 4 (decode steps) = 6
    assert gen_ids[0, 2].item() == 2  # 吐出的 EOS
    assert (gen_ids[0, 3:] == 0).all()  # 陪跑冻结为 PAD


def test_padding_does_not_corrupt_generation():
    """金标准一致性测试：同一个 Prompt 单独跑的结果，与塞入长短不一的 Batch 中跑出的结果 100% 逐字一致。"""
    config = Qwen3Config(
        vocab_size=151646,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(config).eval()

    # Sequence A: 短 prompt [10, 20]
    prompt_a = torch.tensor([[10, 20]], dtype=torch.long)
    gen_single_a, _, _ = generate_tokens(
        model=model,
        input_ids=prompt_a,
        max_new_tokens=5,
        use_cache=True,
    )

    # 构造含 Sequence A 和更长 Sequence B 的批处理
    # Sequence B: [30, 40, 50, 60] (长度 4)
    # Sequence A 左填充 2 个 PAD (假设 PAD ID 为 0): [0, 0, 10, 20]
    batch_input_ids = torch.tensor(
        [
            [0, 0, 10, 20],
            [30, 40, 50, 60],
        ],
        dtype=torch.long,
    )
    batch_mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    gen_batch, _, _ = generate_tokens(
        model=model,
        input_ids=batch_input_ids,
        padding_mask=batch_mask,
        pad_token_id=0,
        max_new_tokens=5,
        use_cache=True,
    )

    # 提取 batch 中序列 A 的生成内容 (从左填充后偏移开始提取)
    # 截取掉前 2 个 PAD，有效内容为 [10, 20, new_tokens...]
    gen_batch_a_unpadded = gen_batch[0, 2:]

    # 验证：单请求生成的全部 Token 与 Batch 中生成的全部 Token 完全严格一致！
    assert torch.equal(gen_single_a[0], gen_batch_a_unpadded), (
        f"Mismatch! Single: {gen_single_a[0].tolist()}, "
        f"Batch: {gen_batch_a_unpadded.tolist()}"
    )


def test_inference_engine_batch_generation(tokenizer):
    """验证 InferenceEngine 的批量文本生成对外接口。"""
    config = Qwen3Config(
        vocab_size=len(tokenizer.tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(config).eval()

    engine = InferenceEngine(model=model, tokenizer=tokenizer)
    prompts = ["Hello", "Explain quantum physics in one sentence"]

    results = engine.generate(prompts, max_new_tokens=4)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["prompt"] == prompts[0]
    assert results[1]["prompt"] == prompts[1]
    assert isinstance(results[0]["generated_text"], str)
    assert isinstance(results[1]["generated_text"], str)
    assert results[0]["token_count"] == 4
    assert results[1]["token_count"] == 4
