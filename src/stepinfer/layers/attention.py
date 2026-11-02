"""带因果掩码 (Causal Mask) 与 GQA (Grouped Query Attention) 支持的纯点积注意力算子实现。"""

import math
from ipykernel.utils import V
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """因果自注意力核心计算算子 (Causal Self-Attention Layer)。

    专注于在给定 Query, Key, Value 张量下的缩放点积注意力与因果掩码计算。
    本层为无状态纯算子层（无持久可学习权重），方便后续无缝替换为 FlashAttention 或 PagedAttention。

    计算公式:
        1. 对齐 GQA 头数: 将 Key 和 Value 的头数通过 repeat 扩展至与 Query 一致
        2. 缩放点积评分: Attention Scores = (Q @ K^T) * scale
        3. 施加因果掩码: 将未来时刻位置置为负无穷 (以防注意力前瞻)
        4. Softmax 概率归一化: Attention Weights = Softmax(Masked Scores, dim=-1)
        5. 上下文加权求和: Output = Attention Weights @ V

    输入与输出维度 (Tensor Shapes):
        - q: [batch_size, seq_len, num_heads, head_dim]
        - k: [batch_size, seq_len, num_kv_heads, head_dim]
        - v: [batch_size, seq_len, num_kv_heads, head_dim]
        - output: [batch_size, seq_len, num_heads, head_dim]
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        scale: float | None = None,
    ) -> None:
        """初始化因果注意力算子超参数。

        参数:
            num_heads (int): Query 的注意力头数。
            head_dim (int): 每个注意力头的特征维度。
            num_kv_heads (int | None): Key 和 Value 的注意力头数。若为 None 则退化为标准 MHA (多头注意力)。
            scale (float | None): 点积缩放因子。若为 None，默认为 1.0 / sqrt(head_dim)。
        """
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"num_heads ({num_heads}) and head_dim ({head_dim}) must be positive integers."
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads ({self.num_kv_heads}) must be a positive integer.")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})."
            )

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """执行因果点积自注意力计算。

        参数:
            q (torch.Tensor): 形状为 [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): 形状为 [batch_size, seq_len, num_kv_heads, head_dim]
            v (torch.Tensor): 形状为 [batch_size, seq_len, num_kv_heads, head_dim]

        返回:
            torch.Tensor: 形状为 [batch_size, seq_len, num_heads, head_dim]
        """
        # 1. 基础维度与形状防御性检查
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"Input tensors must be 4D [batch, seq_len, heads, head_dim], "
                f"got q:{q.ndim}D, k:{k.ndim}D, v:{v.ndim}D."
            )

        batch_size, seq_len, q_heads, q_dim = q.shape
        _, k_seq_len, k_heads, k_dim = k.shape
        _, v_seq_len, v_heads, v_dim = v.shape

        if seq_len != k_seq_len or seq_len != v_seq_len:
            raise ValueError(
                f"Sequence lengths do not match: q={seq_len}, k={k_seq_len}, v={v_seq_len}."
            )

        if q_heads != self.num_heads or k_heads != self.num_kv_heads or v_heads != self.num_kv_heads:
            raise ValueError(
                f"Head count mismatch: expected q_heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
                f"but got q:{q_heads}, k:{k_heads}, v:{v_heads}."
            )

        if q_dim != self.head_dim or k_dim != self.head_dim or v_dim != self.head_dim:
            raise ValueError(
                f"head_dim mismatch: expected {self.head_dim}, got q:{q_dim}, k:{k_dim}, v:{v_dim}."
            )

        # 1. GQA 头数对齐 (若 Query 头数大于 KV 头数)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # 2. 维度转置为 [batch, num_heads, seq_len, head_dim] 以支持批矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. 缩放点积与因果掩码遮蔽
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        # 4. 权重归一化与加权求和
        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, v)

        # 5. 转置回 [batch, seq_len, num_heads, head_dim] 并保持内存连续
        return output.transpose(1, 2).contiguous()
