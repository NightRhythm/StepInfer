"""旋转位置编码 (RoPE, Rotary Position Embedding) 算子实现。"""

import torch
import torch.nn as nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对输入的张量应用旋转位置编码 (Qwen3/LLaMA 规范：Half-Split 前后两半切分旋转)。

    数学公式:
        将输入末尾维度 [..., head_dim] 拆分为前后两半 x1, x2:
            y1 = x1 * cos - x2 * sin
            y2 = x2 * cos + x1 * sin
        拼接回原始维度:
            y = cat([y1, y2], dim=-1)

    参数:
        x: 待旋转张量 (Query 或 Key)，形状为 [batch, seq_len, num_heads, head_dim]
        cos: 余弦权重张量，形状通常为 [batch, seq_len, 1, head_dim // 2]
        sin: 正弦权重张量，形状通常为 [batch, seq_len, 1, head_dim // 2]

    返回:
        旋转后的张量，形状和数据类型与 x 保持一致。
    """
    orig_dtype = x.dtype
    x_fp32 = x.float()

    # Half-Split: 沿末尾维度拆分为前后两半
    half_dim = x_fp32.size(-1) // 2
    x1 = x_fp32[..., :half_dim]
    x2 = x_fp32[..., half_dim:]

    # 确保三角函数权重与输入处于相同设备并在 float32 下计算
    cos_fp32 = cos.to(device=x.device, dtype=torch.float32)
    sin_fp32 = sin.to(device=x.device, dtype=torch.float32)

    # 旋转矩阵计算
    y1 = x1 * cos_fp32 - x2 * sin_fp32
    y2 = x2 * cos_fp32 + x1 * sin_fp32

    # 拼接并还原数据类型
    return torch.cat([y1, y2], dim=-1).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码模块。

    负责在模型初始化阶段预计算全量位置的 cos 和 sin 缓存表 (cos_sin_cache)，
    并在前向传播时根据传入的 Token 位置索引 (positions) 查表并对 Query 和 Key 执行旋转。

    Attributes:
        head_dim: 单个注意力的维度大小 (如 128)。必须为正偶数。
        max_position_embeddings: 支持的最大上下文长度 (默认 32768)。
        base: RoPE 基频 theta (Qwen3 默认为 1000000.0)。
        cos_sin_cache: 预计算好的正余弦联合缓存表，形状为 [max_position_embeddings, head_dim]。
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if base <= 0:
            raise ValueError(f"base (rope_theta) must be positive, got {base}")

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 1. 计算每个旋转平面的逆频率 inv_freq: theta_i = 1.0 / (base ** (2i / head_dim))
        # 形状: [head_dim // 2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

        # 2. 生成所有离散位置索引: [0, 1, 2, ..., max_position_embeddings - 1]
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)

        # 3. 外积计算每一个位置与每一个平面的相角 freqs: [max_position_embeddings, head_dim // 2]
        freqs = torch.outer(t, inv_freq)

        # 4. 计算 cos 和 sin，并合并为单一连续张量以优化显存连续读取访存 (技巧 2)
        # cos: [max_pos, head_dim // 2], sin: [max_pos, head_dim // 2]
        # 合并后 cache: [max_pos, head_dim]
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # 注册为非持久化 buffer (不参与权重保存，但能自动跟随模型移动至对应 GPU/CPU 设备)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向查表并对 Query 和 Key 执行旋转变换。

        输入张量形状 (Tensor Shapes):
            positions: 序列位置索引，支持 1D [seq_len] 或 2D [batch_size, seq_len]
            query: [batch_size, seq_len, num_attention_heads, head_dim]
            key:   [batch_size, seq_len, num_key_value_heads, head_dim]

        返回:
            旋转后的 (query, key)，形状与输入保持完全一致。
        """
        # 1. positions 防御性范围校验
        if (positions < 0).any() or (positions >= self.max_position_embeddings).any():
            raise ValueError(
                f"Position indices out of bounds: must be in [0, {self.max_position_embeddings}), "
                f"got min={positions.min().item()}, max={positions.max().item()}."
            )

        # 2. positions 维度校验
        if positions.dim() not in (1, 2):
            raise ValueError(
                f"positions tensor must be 1D or 2D, got {positions.dim()}D with shape {positions.shape}."
            )

        # 3. 查表获取当前位置的 cos_sin
        cos_sin = self.cos_sin_cache[positions]  # type: ignore
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 4. 增加 head 维度 (unsqueeze(-2)) 适配多头广播
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        # 5. 若 positions 为 1D [seq_len]，额外补齐第 0 维 batch_size 以广播匹配 4D query/key
        if positions.dim() == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # 6. 分别旋转 query 与 key
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)
