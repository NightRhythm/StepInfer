"""Qwen3 核心模型架构实现，包含 MLP、Attention、DecoderLayer 及 ForCausalLM 顶层总装。"""

import torch
import torch.nn as nn

from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.layers.activation import SiluAndMul
from stepinfer.layers.attention import CausalAttention
from stepinfer.layers.norm import RMSNorm
from stepinfer.layers.rotary_embedding import RotaryEmbedding


class Qwen3MLP(nn.Module):
    """Qwen3 SwiGLU 门控前馈网络 (Gated Feed-Forward Network)。

    采用合并投影优化 (Merged gate_up_proj):
        1. 输入 x 经过一次合并投影生成 2 * intermediate_size 的张量
        2. 经过 SiluAndMul 算子进行切半门控相乘: SiLU(gate) * up
        3. 经过 down_proj 降维回 hidden_size
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算。

        参数:
            x (torch.Tensor): 形状为 [batch_size, seq_len, hidden_size]

        返回:
            torch.Tensor: 形状为 [batch_size, seq_len, hidden_size]
        """
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Qwen3Attention(nn.Module):
    """Qwen3 自注意力层 (含 QKV 融合投影、QK-Norm、RoPE 与无状态因果注意力算子)。

    计算流向 (严格时序):
        hidden_states -> qkv_proj -> split(q, k, v) -> reshape
        -> q_norm & k_norm (QK-Norm)
        -> rotary_emb(positions, q, k) (RoPE)
        -> attn(q, k, v) (Causal Self-Attention)
        -> flatten -> o_proj -> output
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 合并 QKV 投影：单次 GEMM 完成所有投射
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        # QK-Norm: 对每个头内部做 RMSNorm (防数值发散)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 旋转位置编码 (RoPE)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        # 纯注意力计算算子
        self.attn = CausalAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )
        # 输出投影
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播计算。

        参数:
            positions (torch.Tensor): 1D [seq_len] 或 2D [batch, seq_len] 的绝对位置索引
            hidden_states (torch.Tensor): 形状为 [batch_size, seq_len, hidden_size]

        返回:
            torch.Tensor: 形状为 [batch_size, seq_len, hidden_size]
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be a 3D tensor [batch, seq_len, hidden_size], "
                f"got {hidden_states.ndim}D."
            )
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 合并投影并切分为 Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. 变换为 4D [batch_size, seq_len, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 3. QK-Norm 归一化 (防数值发散)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 5. 点积因果自注意力
        attn_out = self.attn(q, k, v)

        # 6. 展平并映射输出
        attn_out = attn_out.flatten(2, 3)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 基础解码器层 (Pre-Norm + Fused Residual + Attention + MLP)。

    支持残差融合流水线 (Fused Add + RMSNorm)。
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播计算。

        参数:
            positions (torch.Tensor): 位置索引张量
            hidden_states (torch.Tensor): 输入隐藏状态 [batch, seq_len, hidden_size]
            residual (torch.Tensor | None): 上一层累加传递下来的残差张量

        返回:
            tuple[torch.Tensor, torch.Tensor]:
                - hidden_states: 本层 MLP 计算后的输出
                - residual: 本层累加后的残差张量，继续传递给下一层
        """
        # 1. Pre-Attention RMSNorm + 残差起点初始化
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 2. Self-Attention 前向
        hidden_states = self.self_attn(positions, hidden_states)

        # 3. Post-Attention RMSNorm (融合残差累加)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 4. MLP 前向
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual # type: ignore


class Qwen3Model(nn.Module):
    """Qwen3 Transformer 主干模型 (Embedding + N 个 DecoderLayers + Final RMSNorm)。"""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """模型主干前向传播。

        参数:
            input_ids (torch.Tensor): 输入 Token ID，形状 [batch_size, seq_len]
            positions (torch.Tensor | None): 可选位置索引，为 None 时自动生成 [0, 1, ..., seq_len - 1]

        返回:
            torch.Tensor: 最终归一化后的隐藏状态，形状 [batch_size, seq_len, hidden_size]
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be a 2D tensor [batch_size, seq_len], got {input_ids.ndim}D."
            )

        batch_size, seq_len = input_ids.shape

        # 若未显式传入 positions，默认生成 1D 等差位置张量
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)

        # 1. Token Embedding 查表
        hidden_states = self.embed_tokens(input_ids)

        # 2. 逐层前向传递并累加残差
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 3. 最终层归一化与残差融合
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型顶层类 (主干网络 + LM Head + 权重共享)。"""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享 (Weight Tying): 输出头权重与 Embedding 权重共享内存指针
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """端到端因果语言模型前向计算。

        参数:
            input_ids (torch.Tensor): 输入 Token 序列，形状 [batch_size, seq_len]
            positions (torch.Tensor | None): 可选位置索引

        返回:
            torch.Tensor: 输出预测 Logits，形状 [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits
