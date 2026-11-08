"""KV Cache 核心键值缓存单元测试。

验证预分配显存张量、Prefill 批量写入、Decode 单步追加、切片视图、游标推进、重置及边界防御。
"""

import pytest
import torch

from stepinfer.cache.kv_cache import KVCache, LayerKVCache
from stepinfer.config.qwen3_config import Qwen3Config


def test_layer_kv_cache_initialization():
    """验证 LayerKVCache 初始显存张量分配与属性设置。"""
    max_batch_size = 1
    num_kv_heads = 8
    max_seq_len = 128
    head_dim = 64
    dtype = torch.float32

    cache = LayerKVCache(
        max_batch_size=max_batch_size,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        dtype=dtype,
        device="cpu",
    )

    assert cache.max_batch_size == max_batch_size
    assert cache.num_kv_heads == num_kv_heads
    assert cache.max_seq_len == max_seq_len
    assert cache.head_dim == head_dim
    assert cache.current_length == 0
    assert cache.k_cache.shape == (max_batch_size, num_kv_heads, max_seq_len, head_dim)
    assert cache.v_cache.shape == (max_batch_size, num_kv_heads, max_seq_len, head_dim)
    assert cache.k_cache.dtype == dtype
    assert cache.v_cache.dtype == dtype


def test_layer_kv_cache_prefill_and_decode_step():
    """验证 Prefill 首步长序列写入与 Decode 多步单 Token 逐步切片追加写入。"""
    cache = LayerKVCache(
        max_batch_size=1,
        num_kv_heads=4,
        max_seq_len=32,
        head_dim=16,
    )

    # 1. 模拟 Prefill 首步写入 (序列长度 = 5)
    k_prompt = torch.randn(1, 4, 5, 16)
    v_prompt = torch.randn(1, 4, 5, 16)
    k_out, v_out = cache.update(k_prompt, v_prompt)

    assert cache.current_length == 5
    assert k_out.shape == (1, 4, 5, 16)
    assert v_out.shape == (1, 4, 5, 16)
    assert torch.equal(k_out, k_prompt)
    assert torch.equal(v_out, v_prompt)

    # 2. 模拟 Decode 第 1 步 (追加 1 个 Token)
    k_dec1 = torch.randn(1, 4, 1, 16)
    v_dec1 = torch.randn(1, 4, 1, 16)
    k_out_1, v_out_1 = cache.update(k_dec1, v_dec1)

    assert cache.current_length == 6
    assert k_out_1.shape == (1, 4, 6, 16)
    assert v_out_1.shape == (1, 4, 6, 16)
    # 验证前 5 个 token 历史保持不变，最后一个为新写入
    assert torch.equal(k_out_1[:, :, :5, :], k_prompt)
    assert torch.equal(k_out_1[:, :, 5:, :], k_dec1)
    assert torch.equal(v_out_1[:, :, :5, :], v_prompt)
    assert torch.equal(v_out_1[:, :, 5:, :], v_dec1)

    # 3. 模拟 Decode 第 2 步 (追加第 2 个 Token)
    k_dec2 = torch.randn(1, 4, 1, 16)
    v_dec2 = torch.randn(1, 4, 1, 16)
    k_out_2, v_out_2 = cache.update(k_dec2, v_dec2)

    assert cache.current_length == 7
    assert k_out_2.shape == (1, 4, 7, 16)
    assert v_out_2.shape == (1, 4, 7, 16)
    assert torch.equal(k_out_2[:, :, :5, :], k_prompt)
    assert torch.equal(k_out_2[:, :, 5:6, :], k_dec1)
    assert torch.equal(k_out_2[:, :, 6:, :], k_dec2)


def test_layer_kv_cache_reset():
    """验证 reset 操作能够正确归零游标并使后续写入从 0 开始覆盖。"""
    cache = LayerKVCache(
        max_batch_size=1,
        num_kv_heads=2,
        max_seq_len=16,
        head_dim=8,
    )

    # 先填充 4 个 token
    k1 = torch.randn(1, 2, 4, 8)
    v1 = torch.randn(1, 2, 4, 8)
    cache.update(k1, v1)
    assert cache.current_length == 4

    # 重置
    cache.reset()
    assert cache.current_length == 0

    # 重新写入 2 个新 token
    k2 = torch.randn(1, 2, 2, 8)
    v2 = torch.randn(1, 2, 2, 8)
    k_out, v_out = cache.update(k2, v2)

    assert cache.current_length == 2
    assert k_out.shape == (1, 2, 2, 8)
    assert v_out.shape == (1, 2, 2, 8)
    assert torch.equal(k_out, k2)
    assert torch.equal(v_out, v2)


def test_layer_kv_cache_capacity_overflow():
    """验证输入超出 max_seq_len 时能够正确拦截并抛出 ValueError。"""
    cache = LayerKVCache(
        max_batch_size=1,
        num_kv_heads=2,
        max_seq_len=10,
        head_dim=8,
    )

    # 写入 7 个 token (剩余容量 3)
    k1 = torch.randn(1, 2, 7, 8)
    v1 = torch.randn(1, 2, 7, 8)
    cache.update(k1, v1)

    # 尝试再写入 4 个 token (总长 11 > 10)
    k2 = torch.randn(1, 2, 4, 8)
    v2 = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="KV Cache capacity exceeded"):
        cache.update(k2, v2)

    # 确认溢出失败时游标未被污染推进
    assert cache.current_length == 7


def test_layer_kv_cache_shape_validation():
    """验证 LayerKVCache 的初始化超参及 update 输入张量维度的严格校验。"""
    # 1. 非正初始化参数拦截
    with pytest.raises(ValueError, match="max_batch_size must be a positive integer"):
        LayerKVCache(max_batch_size=0, num_kv_heads=4, max_seq_len=16, head_dim=8)

    with pytest.raises(ValueError, match="num_kv_heads must be a positive integer"):
        LayerKVCache(max_batch_size=1, num_kv_heads=0, max_seq_len=16, head_dim=8)

    with pytest.raises(ValueError, match="max_seq_len must be a positive integer"):
        LayerKVCache(max_batch_size=1, num_kv_heads=4, max_seq_len=-5, head_dim=8)

    with pytest.raises(ValueError, match="head_dim must be a positive integer"):
        LayerKVCache(max_batch_size=1, num_kv_heads=4, max_seq_len=16, head_dim=0)

    # 2. update 输入张量防御拦截
    cache = LayerKVCache(max_batch_size=1, num_kv_heads=4, max_seq_len=16, head_dim=8)

    # 非 Tensor 输入
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        cache.update([1, 2, 3], torch.randn(1, 4, 1, 8))  # type: ignore

    # 非 4D 张量
    with pytest.raises(ValueError, match="must be 4D tensors"):
        cache.update(torch.randn(1, 4, 8), torch.randn(1, 4, 8))

    # k_new 与 v_new 形状不匹配
    with pytest.raises(ValueError, match="Shape mismatch"):
        cache.update(torch.randn(1, 4, 2, 8), torch.randn(1, 4, 3, 8))

    # batch_size 超标
    with pytest.raises(ValueError, match="exceeds max_batch_size"):
        cache.update(torch.randn(2, 4, 1, 8), torch.randn(2, 4, 1, 8))

    # num_kv_heads 不匹配
    with pytest.raises(ValueError, match="num_kv_heads mismatch"):
        cache.update(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))

    # head_dim 不匹配
    with pytest.raises(ValueError, match="head_dim mismatch"):
        cache.update(torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))


def test_kv_cache_container_lifecycle_and_from_config():
    """验证整网全层容器 KVCache 的索引、重置及 from_config 工厂构建。"""
    config = Qwen3Config(
        num_hidden_layers=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )

    cache = KVCache.from_config(config, max_batch_size=1, max_seq_len=32)

    assert len(cache) == 4
    assert isinstance(cache[0], LayerKVCache)
    assert cache[0].num_kv_heads == 2
    assert cache[0].head_dim == 16
    assert cache[0].max_seq_len == 32

    # 索引越界
    with pytest.raises(IndexError, match="out of range"):
        _ = cache[4]
    with pytest.raises(IndexError, match="out of range"):
        _ = cache[-1]

    # 给每一层写入不同长度
    cache[0].update(torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16))
    cache[1].update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16))

    assert cache[0].current_length == 3
    assert cache[1].current_length == 5

    # 全网容器 reset
    cache.reset()
    assert cache[0].current_length == 0
    assert cache[1].current_length == 0
