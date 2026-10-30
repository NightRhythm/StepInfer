"""项目 tokenizer 的统一接口。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTokenizer(ABC):
    """所有 tokenizer 实现都应遵守的最小接口。"""

    @abstractmethod
    def encode(self, text: str | list[str], **kwargs: Any) -> Any:
        """将文本转换为 token IDs 或相关编码结果。"""

        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Any, **kwargs: Any) -> str | list[str]:
        """将 token IDs 转换回文本。"""

        raise NotImplementedError

