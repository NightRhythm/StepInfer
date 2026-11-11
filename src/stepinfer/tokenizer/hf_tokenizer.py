"""基于 Hugging Face AutoTokenizer 的 tokenizer 适配器。"""

from pathlib import Path
from typing import Any

from torch import Tensor
from transformers import AutoTokenizer

from stepinfer.config.model_config import ModelConfig
from stepinfer.tokenizer.base import BaseTokenizer


class HFTokenizer(BaseTokenizer):
    """使用 Hugging Face tokenizer 处理单条文本和 token IDs。"""

    def __init__(self, config: ModelConfig) -> None:
        """根据模型配置初始化 tokenizer。"""

        self.config = config
        self.model_dir: Path = config.model_dir

        # 从本地模型目录加载 tokenizer，不访问远程仓库。
        # 这里的 model_dir 已由 ModelConfig 解析并检查过。
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

    def encode(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = "pt",
        padding: bool | str = False,
        padding_side: str = "left",
        return_attention_mask: bool | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """将单条或批量文本编码为 Token ID (单条) 或 (Token ID, Attention Mask) 元组 (多条)。

        参数:
            text (str | list[str]): 待编码的单条字符串或多条字符串列表
            return_tensors (str | None): 返回格式 ('pt' 为 PyTorch Tensor，None 为 Python 原生列表)
            padding (bool | str): 是否对齐序列长度。传入多条文本时默认为 True
            padding_side (str): 填充方向，默认 'left' (左填充，对齐因果自回归解码末端)
            return_attention_mask (bool | None): 是否返回注意力掩码。默认单条为 False，多条为 True
            truncation (bool): 是否截断超长序列
            max_length (int | None): 截断最大长度

        返回:
            - 单条文本 (str):
                默认返回 input_ids (Tensor [1, seq_len] 或 list[int])；
                若 return_attention_mask=True，返回 (input_ids, attention_mask)。
            - 批量文本 (list[str]):
                默认返回 (input_ids, attention_mask)，形状均为 [batch_size, max_seq_len]；
                若 return_attention_mask=False，返回 input_ids。
        """
        if not isinstance(text, (str, list)):
            raise TypeError(f"text must be a string or a list of strings, got {type(text).__name__}.")

        if isinstance(text, list):
            if len(text) == 0:
                raise ValueError("text list cannot be empty.")
            if not all(isinstance(t, str) for t in text):
                raise TypeError("All items in text list must be strings.")

        if return_tensors not in (None, "pt"):
            raise ValueError("return_tensors must be None or 'pt'.")

        is_batch = isinstance(text, list)
        if return_attention_mask is None:
            return_attention_mask = is_batch

        # 批量编码默认启用左填充
        effective_padding = padding if padding is not False else (True if is_batch else False)

        orig_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = padding_side
            encoded = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=effective_padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
        finally:
            self.tokenizer.padding_side = orig_padding_side

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        if return_attention_mask:
            return input_ids, attention_mask
        return input_ids

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str | list[str]:
        """将单条或批量 Token ID 解码为文本。

        参数:
            token_ids:
                - 单条: 1D Tensor、1D list[int] 或 2D Tensor [1, seq_len]
                - 批量: 2D Tensor [batch_size, seq_len] (batch_size > 1) 或 list[list[int]]
            skip_special_tokens (bool): 是否跳过特殊 Token (如 PAD、EOS)

        返回:
            单条输入返回 str，批量输入返回 list[str]。
        """
        if isinstance(token_ids, Tensor):
            if token_ids.ndim == 1:
                return self.tokenizer.decode(
                    token_ids.tolist(),
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
            elif token_ids.ndim == 2:
                if token_ids.shape[0] == 1:
                    return self.tokenizer.decode(
                        token_ids[0].tolist(),
                        skip_special_tokens=skip_special_tokens,
                        **kwargs,
                    )
                else:
                    return self.tokenizer.batch_decode(
                        token_ids.tolist(),
                        skip_special_tokens=skip_special_tokens,
                        **kwargs,
                    )
            else:
                raise ValueError(
                    f"token_ids tensor must be 1D or 2D, got {token_ids.ndim}D with shape {token_ids.shape}."
                )

        if isinstance(token_ids, list):
            if len(token_ids) == 0:
                return ""
            if all(isinstance(x, int) for x in token_ids):
                return self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
            elif all(isinstance(x, list) for x in token_ids):
                return self.tokenizer.batch_decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
            else:
                raise TypeError("token_ids must be a list of integers or a list of lists of integers.")

        raise TypeError("token_ids must be a torch.Tensor or list.")

    @property
    def eos_token_id(self) -> int | None:
        """返回结束 token 的 ID。"""

        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """返回起始 token 的 ID。"""

        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """返回填充 token 的 ID。"""

        return self.tokenizer.pad_token_id

