"""请求对象与状态机抽象。

定义单条推理请求的元数据、动态生成的 Token 序列、生命周期状态与终止判决。
"""

from enum import Enum
import time


class RequestStatus(str, Enum):
    """请求状态枚举。

    定义单条请求在推理引擎全生命周期中的核心流转状态:
        - WAITING: 已入队，等待调度器分配计算槽位与显存资源；
        - RUNNING: 已被调度器选中，处于 Prefill (首字计算) 或 Decode (迭代生成) 状态；
        - FINISHED: 推理结束 (命中 EOS 或达到最大输出长度)，已移出批次并可回收槽位。
    """

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Request:
    """推理请求核心数据结构。

    纯语义与元数据容器，解耦底层具体 Tensor 与显存，支持增量 Token 记录与状态机迁移。

    核心属性:
        request_id (str): 唯一请求标识符。
        prompt (str): 原始用户输入文本。
        prompt_token_ids (list[int]): 分词后的输入 Token 序列。
        max_new_tokens (int): 当前请求允许生成的新 Token 最大数量。
        arrival_time (float): 请求创建/入队时间戳 (用于 FIFO 排队和延迟统计)。
        status (RequestStatus): 当前请求状态。
        generated_token_ids (list[int]): 自回归生成步中动态产出的 Token 序列。
        stop_reason (str | None): 请求终止原因说明 (如 "stop", "length")。
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        max_new_tokens: int = 50,
        arrival_time: float | None = None,
    ) -> None:
        """初始化推理请求实例并执行防御性校验。

        参数:
            request_id (str): 请求唯一标识符。
            prompt (str): 用户输入字符串。
            prompt_token_ids (list[int]): 编码后的 Prompt Token ID 列表。
            max_new_tokens (int): 生成上限 (必须为正整数)。
            arrival_time (float | None): 到达时间戳，若为 None 则默认取当前时间。

        异常:
            TypeError: 字段类型不符合预期。
            ValueError: request_id 为空、prompt_token_ids 为空或包含负数、max_new_tokens 非正整数。
        """
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id).__name__}.")
        if not request_id.strip():
            raise ValueError("request_id cannot be empty or whitespace-only.")

        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}.")

        if not isinstance(prompt_token_ids, list):
            raise TypeError(
                f"prompt_token_ids must be a list of integers, got {type(prompt_token_ids).__name__}."
            )
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids cannot be empty.")
        for idx, token in enumerate(prompt_token_ids):
            if not isinstance(token, int) or isinstance(token, bool):
                raise TypeError(
                    f"All tokens in prompt_token_ids must be integers, got {type(token).__name__} at index {idx}."
                )
            if token < 0:
                raise ValueError(
                    f"Token ID must be a non-negative integer, got {token} at index {idx}."
                )

        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens}.")

        self.request_id = request_id.strip()
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_new_tokens = max_new_tokens
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # 动态内部状态
        self.status: RequestStatus = RequestStatus.WAITING
        self.generated_token_ids: list[int] = []
        self.stop_reason: str | None = None

    def add_token(self, token_id: int) -> None:
        """追加一个新生成的 Token，并根据生成上限自动触发状态收敛。

        参数:
            token_id (int): 单步生成的 Token ID。

        异常:
            RuntimeError: 请求已处于 FINISHED 状态时禁止追加 Token。
            TypeError: token_id 不是整数。
            ValueError: token_id 为负数。
        """
        if self.is_finished:
            raise RuntimeError(f"Cannot add token to a finished request ({self.request_id}).")

        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(f"token_id must be an integer, got {type(token_id).__name__}.")
        if token_id < 0:
            raise ValueError(f"token_id must be non-negative, got {token_id}.")

        self.generated_token_ids.append(token_id)

        # 达到生成长度上限时，自动流转为 FINISHED
        if len(self.generated_token_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED
            if self.stop_reason is None:
                self.stop_reason = "length"

    def mark_finished(self, reason: str = "stop") -> None:
        """显式将请求标记为完成态 (如命中 EOS 或被外部取消)。

        参数:
            reason (str): 终止原因说明。
        """
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
            f"Request(id={self.request_id!r}, status={self.status.value}, "
            f"prompt_len={self.num_prompt_tokens}, generated_len={self.num_generated_tokens}, "
            f"max_new={self.max_new_tokens})"
        )
