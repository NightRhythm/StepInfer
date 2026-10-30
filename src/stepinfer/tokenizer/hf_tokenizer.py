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

    # TODO: 增加 Batch 支持
    def encode(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = "pt",
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """将单条文本编码为 token IDs。"""

        if not isinstance(text, str):
            raise TypeError("HFTokenizer currently supports a single string only.")

        if return_tensors not in (None, "pt"):
            raise ValueError("return_tensors must be None or 'pt'.")

        if padding:
            raise ValueError("Padding is not supported for single-text encoding.")

        # 直接调用 tokenizer 可以统一处理截断、特殊 token 和 Tensor 输出。
        encoded = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=False,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        input_ids = encoded["input_ids"]
        if return_tensors == "pt":
            return input_ids

        # return_tensors=None 时，单条文本已经是 list[int]，无需再去掉维度。
        return input_ids

    # TODO: 增加 Batch 支持
    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """将一维 token IDs 解码为文本。"""

        if isinstance(token_ids, Tensor):
            if token_ids.ndim == 2 and token_ids.shape[0] == 1:
                token_ids = token_ids[0]
            elif token_ids.ndim != 1:
                raise ValueError(
                    "token_ids must be a 1D tensor or a 2D tensor with batch size 1."
                )
            token_ids = token_ids.tolist()

        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers or a 1D tensor.")

        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise TypeError("token_ids must contain integers only.")

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

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

