"""KV Cache 核心键值缓存单元测试。

验证预分配显存张量、槽位隔离切片写入、独立游标推进、槽位重置与容量溢出边界拦截。
"""

import pytest
import torch

from stepinfer.cache.kv_cache import KVCache, LayerKVCache
from stepinfer.config.qwen3_config import Qwen3Config


def test_layer_kv_cache_initialization():
    """验证 LayerKVCache 初始显存张量分配与独立游标初始化。"""
    max_batch_size = 4
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
    assert cache.seq_lens == [0] * max_batch_size
    assert cache.k_cache.shape == (max_batch_size, num_kv_heads, max_seq_len, head_dim)
    assert cache.v_cache.shape == (max_batch_size, num_kv_heads, max_seq_len, head_dim)


def test_layer_kv_cache_slot_isolation_and_update():
    """验证不同槽位写入不同长度序列，游标互不干扰，且返回统一 4D 历史切片。"""
    cache = LayerKVCache(
        max_batch_size=4,
        num_kv_heads=2,
        max_seq_len=32,
        head_dim=8,
    )

    # 1. 槽位 0 执行 Prefill 首步写入 (长度 = 5)
    k_0 = torch.randn(1, 2, 5, 8)
    v_0 = torch.randn(1, 2, 5, 8)
    k_out_0, v_out_0 = cache.update(k_0, v_0, slot_ids=[0])

    assert cache.get_seq_len(0) == 5
    assert cache.get_seq_len(1) == 0
    assert k_out_0.shape == (1, 2, 5, 8)
    assert torch.equal(k_out_0, k_0)

    # 2. 槽位 2 执行 Prefill 首步写入 (长度 = 2)
    k_2 = torch.randn(1, 2, 2, 8)
    v_2 = torch.randn(1, 2, 2, 8)
    k_out_2, v_out_2 = cache.update(k_2, v_2, slot_ids=[2])

    assert cache.get_seq_len(0) == 5
    assert cache.get_seq_len(2) == 2
    assert k_out_2.shape == (1, 2, 2, 8)

    # 3. 槽位 0 与 槽位 2 同时执行 1 步 Decode 解码 (各追加 1 个 Token)
    k_dec = torch.randn(2, 2, 1, 8)
    v_dec = torch.randn(2, 2, 1, 8)
    # 当前活跃槽位: slot_ids = [0, 2]
    k_out, v_out = cache.update(k_dec, v_dec, slot_ids=[0, 2])

    # 游标各自独立 +1
    assert cache.get_seq_len(0) == 6
    assert cache.get_seq_len(2) == 3
    # 截取最大有效长度 max(6, 3) = 6，输出形状规整为 [2, 2, 6, 8]
    assert k_out.shape == (2, 2, 6, 8)
    assert v_out.shape == (2, 2, 6, 8)

    # 校验槽位 0 历史前 5 个未被破坏，第 6 个是新追加的 k_dec[0]
    assert torch.equal(k_out[0, :, :5, :], k_0[0])
    assert torch.equal(k_out[0, :, 5:6, :], k_dec[0])
    # 校验槽位 2 历史前 2 个未被破坏，第 3 个是新追加的 k_dec[1]
    assert torch.equal(k_out[1, :, :2, :], k_2[0])
    assert torch.equal(k_out[1, :, 2:3, :], k_dec[1])


def test_layer_kv_cache_reset_slot():
    """验证单槽位 reset_slot 仅清零目标槽位，其他槽位数据完好无损且支持复用。"""
    cache = LayerKVCache(
        max_batch_size=3,
        num_kv_heads=2,
        max_seq_len=16,
        head_dim=8,
    )

    # 填充槽位 0 (长 4) 与 槽位 1 (长 3)
    k_0 = torch.randn(1, 2, 4, 8)
    v_0 = torch.randn(1, 2, 4, 8)
    cache.update(k_0, v_0, slot_ids=[0])

    k_1 = torch.randn(1, 2, 3, 8)
    v_1 = torch.randn(1, 2, 3, 8)
    cache.update(k_1, v_1, slot_ids=[1])

    # 仅重置槽位 0
    cache.reset_slot(0)
    assert cache.get_seq_len(0) == 0
    assert torch.all(cache.k_cache[0] == 0)
    # 槽位 1 完好无损
    assert cache.get_seq_len(1) == 3
    assert torch.equal(cache.k_cache[1, :, :3, :], k_1[0])

    # 槽位 0 复用给新序列 (长 2)
    k_new = torch.randn(1, 2, 2, 8)
    v_new = torch.randn(1, 2, 2, 8)
    k_out, v_out = cache.update(k_new, v_new, slot_ids=[0])

    assert cache.get_seq_len(0) == 2
    assert torch.equal(k_out[0], k_new[0])


def test_layer_kv_cache_capacity_overflow():
    """验证指定槽位写入超出 max_seq_len 时正确拦截。"""
    cache = LayerKVCache(
        max_batch_size=2,
        num_kv_heads=2,
        max_seq_len=10,
        head_dim=8,
    )

    k_init = torch.randn(1, 2, 8, 8)
    v_init = torch.randn(1, 2, 8, 8)
    cache.update(k_init, v_init, slot_ids=[0])

    # 槽位 0 剩余空间仅 2，尝试写入 3 个 token
    k_over = torch.randn(1, 2, 3, 8)
    v_over = torch.randn(1, 2, 3, 8)
    with pytest.raises(ValueError, match="KV Cache capacity exceeded for slot 0"):
        cache.update(k_over, v_over, slot_ids=[0])


def test_layer_kv_cache_slot_bounds_validation():
    """验证槽位与批大小不一致、槽位越界的拦截。"""
    cache = LayerKVCache(max_batch_size=2, num_kv_heads=2, max_seq_len=16, head_dim=8)

    k = torch.randn(2, 2, 1, 8)
    v = torch.randn(2, 2, 1, 8)

    # slot_ids 数量与 batch_size 不符
    with pytest.raises(ValueError, match="Batch size mismatch"):
        cache.update(k, v, slot_ids=[0])

    # slot_id 越界
    with pytest.raises(IndexError, match="out of bounds"):
        cache.update(k, v, slot_ids=[0, 5])


def test_kv_cache_container_lifecycle():
    """验证全网容器 KVCache 实例的穿透槽位管理与全局重置。"""
    config = Qwen3Config(
        num_hidden_layers=3,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )

    cache = KVCache.from_config(config, max_batch_size=2, max_seq_len=32)

    assert len(cache) == 3
    assert isinstance(cache[0], LayerKVCache)

    # 写入槽位 1
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    cache[0].update(k, v, slot_ids=[1])
    cache[1].update(k, v, slot_ids=[1])

    assert cache.get_seq_len(1) == 4

    # 穿透清空槽位 1
    cache.reset_slot(1)
    assert cache.get_seq_len(1) == 0

    # 全网全局 reset
    cache.reset()
    assert cache.get_seq_len(0) == 0
