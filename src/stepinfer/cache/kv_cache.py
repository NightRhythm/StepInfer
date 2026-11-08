"""KV Cache 核心键值缓存模块。

提供按层预分配连续显存张量、原地更新切片与有效历史视图切分的管理机制。
"""

from typing import Any

import torch
from torch import Tensor


class LayerKVCache:
    """单层 Transformer 解码器的静态预分配 KV Cache。

    核心设计考量:
        1. 显存预分配: 在模型推理前一次性分配最大上下文长度 (max_seq_len) 的显存，
           避免推理过程中频繁调用 torch.cat 或动态分配导致显存碎片与开销；
        2. 原地切片写入 (In-place update): 每次推理只需将最新步的 k_new, v_new
           写入底层 buffer 的 [current_len : current_len + new_len] 槽位；
        3. GQA 原生保存: 仅保存未被 repeat 广播的原生 num_kv_heads，减少 50% 以上显存占用；
        4. 零拷贝视图返回: 返回截止到当前有效长度 [:current_len] 的 tensor 切片视图。

    张量形状约定 (Tensor Contract):
        - 预分配底层缓存: [batch_size, num_kv_heads, max_seq_len, head_dim]
        - 待写入新键值 (k_new / v_new): [batch_size, num_kv_heads, new_seq_len, head_dim]
        - 返回有效历史键值 (k_out / v_out): [batch_size, num_kv_heads, current_len, head_dim]
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
            max_batch_size (int): 支持的最大批大小 (当前阶段固定为 1)。
            num_kv_heads (int): Key / Value 的头数 (如 Qwen3 为 8)。
            max_seq_len (int): 支持的最大序列长度 (如 2048 / 32768)。
            head_dim (int): 单头特征维度 (如 128)。
            dtype (torch.dtype): 缓存张量数据类型。
            device (torch.device | str): 缓存分配的设备 (如 "cpu" 或 "cuda")。
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

        # 当前已缓存的有效序列长度指针 (游标)
        self.current_len: int = 0

        # 预分配连续张量显存 [batch_size, num_kv_heads, max_seq_len, head_dim]
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

    def update(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """将当前步新计算的 k_new, v_new 原地写入缓存，并返回有效历史切片。

        参数:
            k_new (Tensor): 新输入的 Key 张量，形状为 [batch_size, num_kv_heads, seq_len, head_dim]
            v_new (Tensor): 新输入的 Value 张量，形状为 [batch_size, num_kv_heads, seq_len, head_dim]

        返回:
            tuple[Tensor, Tensor]: 包含历史所有有效 token 的 (k_out, v_out)，
            形状均为 [batch_size, num_kv_heads, current_len, head_dim]。

        异常:
            TypeError: 输入不是 torch.Tensor。
            ValueError: 张量维度、头数、head_dim 不匹配，或序列长度超过 max_seq_len。
        """
        # 1. 防御性类型与基础维度检查
        if not isinstance(k_new, Tensor) or not isinstance(v_new, Tensor):
            raise TypeError(
                f"k_new and v_new must be torch.Tensor, got k:{type(k_new)}, v:{type(v_new)}."
            )

        if k_new.ndim != 4 or v_new.ndim != 4:
            raise ValueError(
                f"k_new and v_new must be 4D tensors [batch, num_kv_heads, seq_len, head_dim], "
                f"got k:{k_new.ndim}D, v:{v_new.ndim}D."
            )

        if k_new.shape != v_new.shape:
            raise ValueError(
                f"Shape mismatch between k_new {k_new.shape} and v_new {v_new.shape}."
            )

        batch_size, num_heads, new_seq_len, head_dim = k_new.shape

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )
        if num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_kv_heads mismatch: expected {self.num_kv_heads}, got {num_heads}."
            )
        if head_dim != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: expected {self.head_dim}, got {head_dim}."
            )

        # 2. 边界检查: 严防超出最大容量
        if self.current_len + new_seq_len > self.max_seq_len:
            raise ValueError(
                f"KV Cache capacity exceeded: current_len={self.current_len}, "
                f"incoming_len={new_seq_len}, max_seq_len={self.max_seq_len}."
            )

        # 3. 原地切片更新与有效历史视图切分
        start_pos = self.current_len
        end_pos = start_pos + new_seq_len
        self.k_cache[:batch_size, :, start_pos:end_pos, :] = k_new
        self.v_cache[:batch_size, :, start_pos:end_pos, :] = v_new
        self.current_len = end_pos
        return (
            self.k_cache[:batch_size, :, :end_pos, :],
            self.v_cache[:batch_size, :, :end_pos, :],
        )

    def reset(self) -> None:
        """重置当前层的缓存状态。"""
        self.current_len = 0

    @property
    def current_length(self) -> int:
        """获取当前已缓存的有效序列长度。"""
        return self.current_len


class KVCache:
    """整网全层 KV Cache 容器。

    管理 Transformer 架构下所有 Decoder 层的 LayerKVCache 实例。
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
            num_layers (int): Transformer 模型层数 (如 28 层)。
            max_batch_size (int): 支持的最大批大小。
            num_kv_heads (int): Key / Value 头数。
            max_seq_len (int): 最大序列长度。
            head_dim (int): 注意力头维度。
            dtype (torch.dtype): 缓存张量数据类型。
            device (torch.device | str): 分配设备。
        """
        if num_layers <= 0:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers}.")

        self.num_layers = num_layers
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

    def reset(self) -> None:
        """清空所有层的缓存游标。"""
        for layer in self.layers:
            layer.reset()

    @classmethod
    def from_config(
        cls,
        config: Any,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> "KVCache":
        """便捷工厂方法：直接从模型配置类 (如 Qwen3Config) 构建全层 KVCache 实例。

        参数:
            config: 包含 num_hidden_layers, num_key_value_heads, head_dim 等属性的配置对象。
            max_batch_size (int): 批大小。
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
