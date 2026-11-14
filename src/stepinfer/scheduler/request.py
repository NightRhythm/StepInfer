"""请求对象与状态机抽象。

定义单条推理请求的元数据、动态生成的 Token 序列、生命周期状态与终止判决。
"""

from enum import Enum
import time


class RequestStatus(str, Enum):
    """请求状态枚举。"""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Request:
    """推理请求核心数据结构。

    纯语义与元数据容器，解耦底层具体 Tensor 与显存，支持增量 Token 记录与状态机迁移。
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        max_new_tokens: int = 50,
        arrival_time: float | None = None,
        slot_id: int | None = None,
    ) -> None:
        """初始化推理请求实例。"""
        if not request_id or not request_id.strip():
            raise ValueError("request_id cannot be empty or whitespace-only.")
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids cannot be empty.")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens}.")

        self.request_id = request_id.strip()
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_new_tokens = max_new_tokens
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        self.slot_id = slot_id

        # 动态内部状态
        self.status: RequestStatus = RequestStatus.WAITING
        self.generated_token_ids: list[int] = []
        self.stop_reason: str | None = None

    def add_token(self, token_id: int) -> None:
        """追加一个新生成的 Token，并根据生成上限自动触发状态收敛。"""
        if self.is_finished:
            raise RuntimeError(f"Cannot add token to a finished request ({self.request_id}).")

        self.generated_token_ids.append(token_id)

        # 达到生成长度上限时，自动流转为 FINISHED
        if len(self.generated_token_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED
            if self.stop_reason is None:
                self.stop_reason = "length"

    def mark_finished(self, reason: str = "stop") -> None:
        """显式将请求标记为完成态 (如命中 EOS 或被外部取消)。"""
        self.status = RequestStatus.FINISHED
        self.stop_reason = reason

    @property
    def is_finished(self) -> bool:
        """判断请求是否已经结束。"""
        return self.status == RequestStatus.FINISHED

    @property
    def is_prefill(self) -> bool:
        """判断请求当前是否处于 Prefill 首步阶段 (尚未生成任何新 Token)。"""
        return len(self.generated_token_ids) == 0

    @property
    def num_prompt_tokens(self) -> int:
        """输入 Prompt Token 序列长度。"""
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        """当前已生成的新 Token 数量。"""
        return len(self.generated_token_ids)

    @property
    def total_len(self) -> int:
        """当前请求总序列长度 (Prompt 长度 + 已生成长度)。"""
        return self.num_prompt_tokens + self.num_generated_tokens

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id!r}, status={self.status.value}, slot={self.slot_id}, "
            f"prompt_len={self.num_prompt_tokens}, generated_len={self.num_generated_tokens}, "
            f"max_new={self.max_new_tokens})"
        )
