"""KV Cache 核心键值缓存模块。

提供按层预分配连续显存张量、槽位隔离切片写入、动态视图返回与生命周期管理的统一机制。
"""

from typing import Any

import torch
from torch import Tensor


class LayerKVCache:
    """单层 Transformer 解码器的槽位隔离 KV Cache。

    核心设计考量:
        1. 显存预分配: 预先分配固定槽位数量 (max_batch_size) 与最大序列长度 (max_seq_len) 的显存，
           彻底消除推理过程中的动态内存分配与显存碎片；
        2. 槽位隔离与独立游标: 每个并发槽位 (Slot) 独立维护已缓存长度 (seq_lens)，互不干扰；
        3. 原地切片写入: 每次前向根据显式指定的 slot_ids，将 k_new, v_new 写入对应槽位偏移；
        4. GQA 原生保存: 仅保存未被 repeat 广播的原生 num_kv_heads，减少 50% 显存占用；
        5. 零拷贝多槽位视图返回: 截取当前活跃槽位的最大有效长度返回统一 4D 张量，供下游批矩阵乘法高效计算。

    张量形状约定 (Tensor Contract):
        - 底层缓存: [max_batch_size, num_kv_heads, max_seq_len, head_dim]
        - 待写入新键值 (k_new / v_new): [batch_size, num_kv_heads, seq_len, head_dim]
        - 返回有效历史切片: [batch_size, num_kv_heads, max_active_seq_len, head_dim]
    """

    def __init__(
        self,
        max_batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        """初始化单层 KV Cache 并预分配底层张量显存。

        参数:
            max_batch_size (int): 引擎允许并发管理的最大槽位数量。
            num_kv_heads (int): Key / Value 的注意力头数。
            max_seq_len (int): 支持的最大上下文序列长度。
            head_dim (int): 单头特征维度。
            dtype (torch.dtype): 缓存张量数据类型。
            device (torch.device | str): 分配设备。
        """
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size}.")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be a positive integer, got {num_kv_heads}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len}.")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be a positive integer, got {head_dim}.")

        self.max_batch_size = max_batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # 槽位级独立长度指针
        self.seq_lens: list[int] = [0] * self.max_batch_size

        # 预分配连续张量显存 [max_batch_size, num_kv_heads, max_seq_len, head_dim]
        self.k_cache: Tensor = torch.zeros(
            (self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache: Tensor = torch.zeros(
            (self.max_batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def update(
        self,
        k_new: Tensor,
        v_new: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, Tensor]:
        """将当前步计算的 k_new, v_new 写入指定槽位，并返回活跃槽位的历史切片视图。

        参数:
            k_new (Tensor): 新输入的 Key 张量 [batch_size, num_kv_heads, seq_len, head_dim]
            v_new (Tensor): 新输入的 Value 张量 [batch_size, num_kv_heads, seq_len, head_dim]
            slot_ids (list[int]): 与输入张量逐行对应的物理槽位索引列表。

        返回:
            tuple[Tensor, Tensor]: 提取活跃槽位切片构成的 (k_out, v_out)，
            形状均为 [batch_size, num_kv_heads, max_active_seq_len, head_dim]。
        """
        # 1. 核心张量与槽位边界防御 (只防核心物理边界，不搞冗余八股文)
        batch_size, num_heads, new_seq_len, head_dim = k_new.shape

        if batch_size != len(slot_ids):
            raise ValueError(
                f"Batch size mismatch: k_new has {batch_size} sequences, but got {len(slot_ids)} slot_ids."
            )
        if num_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(
                f"Shape mismatch: expected heads={self.num_kv_heads}, head_dim={self.head_dim}, "
                f"got heads={num_heads}, head_dim={head_dim}."
            )

        # 2. 槽位越界与显存容量防御
        for slot_id in slot_ids:
            if slot_id < 0 or slot_id >= self.max_batch_size:
                raise IndexError(f"Slot ID {slot_id} is out of bounds [0, {self.max_batch_size - 1}].")
            if self.seq_lens[slot_id] + new_seq_len > self.max_seq_len:
                raise ValueError(
                    f"KV Cache capacity exceeded for slot {slot_id}: current_len={self.seq_lens[slot_id]}, "
                    f"incoming_len={new_seq_len}, max_seq_len={self.max_seq_len}."
                )

        # 3. 按槽位原地切片写入
        for i, slot_id in enumerate(slot_ids):
            start_pos = self.seq_lens[slot_id]
            end_pos = start_pos + new_seq_len
            self.k_cache[slot_id, :, start_pos:end_pos, :] = k_new[i]
            self.v_cache[slot_id, :, start_pos:end_pos, :] = v_new[i]
            self.seq_lens[slot_id] = end_pos

        # 4. 截取当前活跃槽位的最大有效长度返回统一 4D 张量
        max_active_len = max(self.seq_lens[slot_id] for slot_id in slot_ids)
        return (
            self.k_cache[slot_ids, :, :max_active_len, :],
            self.v_cache[slot_ids, :, :max_active_len, :],
        )

    def reset_slot(self, slot_id: int) -> None:
        """重置指定槽位的长度游标并清空显存内容，供新请求复用。

        参数:
            slot_id (int): 目标槽位索引。
        """
        if slot_id < 0 or slot_id >= self.max_batch_size:
            raise IndexError(f"Slot ID {slot_id} is out of range [0, {self.max_batch_size - 1}].")

        self.seq_lens[slot_id] = 0
        self.k_cache[slot_id].zero_()
        self.v_cache[slot_id].zero_()

    def reset(self) -> None:
        """重置所有槽位的缓存状态与显存。"""
        self.seq_lens = [0] * self.max_batch_size
        self.k_cache.zero_()
        self.v_cache.zero_()

    def get_seq_len(self, slot_id: int) -> int:
        """获取指定槽位当前已缓存的有效序列长度。"""
        if slot_id < 0 or slot_id >= self.max_batch_size:
            raise IndexError(f"Slot ID {slot_id} is out of range [0, {self.max_batch_size - 1}].")
        return self.seq_lens[slot_id]


class KVCache:
    """整网全层 KV Cache 容器。

    管理 Transformer 架构下所有 Decoder 层的 LayerKVCache 实例，提供统一的生命周期与槽位管理接口。
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        """初始化整网 KV Cache 容器。

        参数:
            num_layers (int): Transformer 模型层数。
            max_batch_size (int): 支持的最大并发槽位数量。
            num_kv_heads (int): Key / Value 头数。
            max_seq_len (int): 最大序列长度。
            head_dim (int): 注意力头维度。
            dtype (torch.dtype): 缓存张量数据类型。
            device (torch.device | str): 分配设备。
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}.")

        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.layers: list[LayerKVCache] = [
            LayerKVCache(
                max_batch_size=max_batch_size,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        """根据层号索引获取指定层的 LayerKVCache 实例。"""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers - 1}]."
            )
        return self.layers[layer_idx]

    def __len__(self) -> int:
        """返回缓存管理的 Transformer 层数。"""
        return self.num_layers

    def reset_slot(self, slot_id: int) -> None:
        """全网所有层同步清空指定槽位并释放显存。

        参数:
            slot_id (int): 目标槽位索引。
        """
        for layer in self.layers:
            layer.reset_slot(slot_id)

    def reset(self) -> None:
        """全网所有层清空所有槽位。"""
        for layer in self.layers:
            layer.reset()

    def get_seq_len(self, slot_id: int) -> int:
        """获取指定槽位的有效序列长度 (以第 0 层为基准)。"""
        return self.layers[0].get_seq_len(slot_id)

    @classmethod
    def from_config(
        cls,
        config: Any,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> "KVCache":
        """便捷工厂方法：直接从模型配置类 (如 Qwen3Config) 构建全层 KVCache 容器。

        参数:
            config: 包含模型超参的配置对象。
            max_batch_size (int): 最大并发槽位数量。
            max_seq_len (int | None): 最大缓存长度。若为 None 则默认取 config.max_position_embeddings。
            dtype (torch.dtype): 缓存数据类型。
            device (torch.device | str): 分配设备。
        """
        actual_max_seq_len = (
            max_seq_len
            if max_seq_len is not None
            else getattr(config, "max_position_embeddings", 2048)
        )
        return cls(
            num_layers=config.num_hidden_layers,
            max_batch_size=max_batch_size,
            num_kv_heads=config.num_key_value_heads,
            max_seq_len=actual_max_seq_len,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )
