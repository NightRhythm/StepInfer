"""Qwen3 模型配置类定义与本地 config.json 参数映射解析。"""

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class Qwen3Config:
    """Qwen3 模型结构超参数配置类。

    Attributes (默认 Qwen3-0.6B-Base 配置):
        vocab_size: 词表大小 (默认 151936)
        hidden_size: 隐藏层维度 (默认 1024)
        intermediate_size: MLP 升维维度 (默认 3072)
        num_hidden_layers: Transformer Decoder 层数 (默认 28)
        num_attention_heads: Query 注意力头数 (默认 16)
        num_key_value_heads: Key/Value 头数，用于 GQA 分组 (默认 8)
        head_dim: 单个注意力的维度大小 (默认 128)
        rms_norm_eps: RMSNorm 算子防除零 epsilon (默认 1e-6)
        rope_theta: RoPE 旋转位置编码的基频 base (默认 1000000.0)
        max_position_embeddings: 模型支持的最大上下文长度 (默认 32768)
        tie_word_embeddings: 输出 LM Head 是否与输入 Embedding 共享权重 (默认 True)
        bos_token_id: 序列开始 token ID (默认 151643)
        eos_token_id: 序列结束 token ID (默认 151643)
    """

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151643

    def __post_init__(self) -> None:
        """配置合法性与维度防御性校验。"""
        # 1. head_dim 处理：若未指定或 <= 0，自动按 hidden_size // num_attention_heads 回退计算
        if self.head_dim is None or self.head_dim <= 0:
            if self.hidden_size <= 0 or self.num_attention_heads <= 0:
                raise ValueError(
                    f"无法推导 head_dim: hidden_size ({self.hidden_size}) 和 "
                    f"num_attention_heads ({self.num_attention_heads}) 必须为正整数。"
                )
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) 必须能被 "
                    f"num_attention_heads ({self.num_attention_heads}) 整除！"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

        # 2. 基础数值大于 0 的校验
        positive_fields = [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
        ]
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"Config error: '{field_name}' must be > 0, got {value}.")

        # 3. 特殊 token ID 范围校验 [0, vocab_size)
        if self.bos_token_id < 0 or self.bos_token_id >= self.vocab_size:
            raise ValueError(
                f"Config error: bos_token_id must be >= 0 and < {self.vocab_size} (vocab_size), got {self.bos_token_id}."
            )
        if self.eos_token_id < 0 or self.eos_token_id >= self.vocab_size:
            raise ValueError(
                f"Config error: eos_token_id must be >= 0 and < {self.vocab_size} (vocab_size), got {self.eos_token_id}."
            )

        # 4. GQA 分组整除性校验 (Query 头数必须能被 Key/Value 头数整除)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"GQA 配置错误: num_attention_heads ({self.num_attention_heads}) "
                f"必须能被 num_key_value_heads ({self.num_key_value_heads}) 整除！"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3Config":
        """从字典解析已知字段，过滤掉无关参数。"""
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_json_file(cls, json_path: str | Path) -> "Qwen3Config":
        """从本地 config.json 文件构建 Qwen3Config 实例。"""
        path = Path(json_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)
