"""调度中枢与批处理控制器。

负责在每个 Token 迭代步进行请求挑选、槽位配额计算、分阶段调度组装与已完成请求的及时回收。
"""

from typing import Any
import torch

from stepinfer.scheduler.request import Request, RequestStatus
from stepinfer.scheduler.request_queue import RequestQueue


class ScheduleResult:
    """调度决策产物。

    封装当前单步迭代决策的输出元数据，告知执行引擎送入模型的候选请求实体及执行阶段。

    核心属性:
        requests (list[Request]): 当前步被调度选中的请求列表。
        is_prefill (bool): 标识当前步是 Prefill (首字全量计算) 还是 Decode (单步自回归解码)。
        num_tokens (int): 当前调度批次预计计算的 Token 总量 (用于吞吐量统计与显存预算核算)。
    """

    def __init__(
        self,
        requests: list[Request],
        is_prefill: bool,
        num_tokens: int = 0,
    ) -> None:
        """初始化调度结果。

        参数:
            requests (list[Request]): 调度的请求列表。
            is_prefill (bool): 是否为 Prefill 阶段。
            num_tokens (int): 涉及的 Token 数量。
        """
        self.requests = list(requests)
        self.is_prefill = is_prefill
        self.num_tokens = num_tokens

    @property
    def is_empty(self) -> bool:
        """判断当前调度结果是否为空 (无需模型前向计算)。"""
        return len(self.requests) == 0

    @property
    def batch_size(self) -> int:
        """当前批次包含的请求数量。"""
        return len(self.requests)

    def __repr__(self) -> str:
        return (
            f"ScheduleResult(batch_size={self.batch_size}, is_prefill={self.is_prefill}, "
            f"num_tokens={self.num_tokens})"
        )


class Scheduler:
    """推理引擎核心调度器。

    控制中心，向下管理等待队列与运行中的请求状态机，实现小步迭代级调度 (Iteration-level Scheduling)。

    核心功能:
        1. 资源与配额管控: 依据 max_batch_size 与 max_seq_len 严格拦截越界与超载；
        2. 分阶段解耦调度: 区分处理 Prefill (长 Prompt 批次) 与 Decode (单 Token 批次)；
        3. 零泄漏生命周期管理: 在单步结束时将完成的请求当场弹出，回收计算槽位。
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        request_queue: RequestQueue | None = None,
    ) -> None:
        """初始化调度器实例。

        参数:
            max_batch_size (int): 引擎允许并发运行的最大请求数量上限。
            max_seq_len (int): 模型支持的最大序列长度上限。
            request_queue (RequestQueue | None): 等待请求队列。若为 None 则默认创建无界队列。

        异常:
            ValueError: max_batch_size 或 max_seq_len 非正整数。
        """
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size}.")

        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len}.")

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.request_queue = request_queue if request_queue is not None else RequestQueue()

        # 运行中的请求集合，使用 dict 保证 O(1) 的按 ID 查找、更新与剔除
        self.running_requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """向调度器提交新请求。

        参数:
            request (Request): 待处理的请求对象。

        异常:
            ValueError: 请求 Prompt 长度超出模型 max_seq_len 上限。
        """
        if request.num_prompt_tokens > self.max_seq_len:
            raise ValueError(
                f"Prompt length ({request.num_prompt_tokens}) exceeds model max_seq_len ({self.max_seq_len})."
            )

        self.request_queue.add_request(request)

    def schedule(self) -> ScheduleResult:
        """执行单步调度裁决，组织当前步的执行批次。

        调度决策逻辑:
            1. 优先满足 Decode: 若已有处于 RUNNING 状态的请求，直接将它们组织成单步解码批次 (is_prefill=False)；
            2. 运行池空闲时推进 Prefill: 若当前无运行中请求且等待队列非空，根据空闲槽位从 WAITING 队列挑选新请求，
               状态切为 RUNNING，组织成首字计算批次 (is_prefill=True)；
            3. 队列全空或槽位已满: 返回空结果 ScheduleResult(is_empty=True)。

        返回:
            ScheduleResult: 调度决策元数据对象。
        """
        # 防御性清理：先检查是否有已被标记为完成的请求残留
        finished_ids = [req_id for req_id, req in self.running_requests.items() if req.is_finished]
        for req_id in finished_ids:
            del self.running_requests[req_id]

        # 策略 1: 优先调度 Decode 批次 (全员单步前进)
        if len(self.running_requests) > 0:
            active_requests = list(self.running_requests.values())
            return ScheduleResult(
                requests=active_requests,
                is_prefill=False,
                num_tokens=len(active_requests),
            )

        # 策略 2: 运行池空闲，从队列拉取新请求进行 Prefill
        available_slots = self.max_batch_size - len(self.running_requests)
        if available_slots <= 0 or self.request_queue.is_empty():
            return ScheduleResult(requests=[], is_prefill=False, num_tokens=0)

        new_requests = self.request_queue.get_waiting_requests(max_requests=available_slots)
        valid_requests: list[Request] = []

        for req in new_requests:
            if req.num_prompt_tokens > self.max_seq_len:
                # 超长请求拦截并标记完成
                req.mark_finished("Prompt length exceeds max_seq_len.")
                continue

            req.status = RequestStatus.RUNNING
            self.running_requests[req.request_id] = req
            valid_requests.append(req)

        if not valid_requests:
            return ScheduleResult(requests=[], is_prefill=False, num_tokens=0)

        total_prompt_tokens = sum(req.num_prompt_tokens for req in valid_requests)
        return ScheduleResult(
            requests=valid_requests,
            is_prefill=True,
            num_tokens=total_prompt_tokens,
        )

    def post_step(
        self,
        requests: list[Request] | None = None,
        next_tokens: list[int] | dict[str, int] | torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> list[Request]:
        """单步执行完毕后的状态写回与槽位回收 (打扫战场)。

        参数:
            requests (list[Request] | None): 当前步参与前向的请求列表。若为 None 则默认取当前所有 running_requests。
            next_tokens (list[int] | dict[str, int] | torch.Tensor | None): 模型输出的新 Token。
            eos_token_id (int | None): 结束符 ID。

        返回:
            list[Request]: 本步刚触发完成 (FINISHED) 的请求列表，可直接交付上层返回给客户端。
        """
        target_requests = requests if requests is not None else list(self.running_requests.values())
        if not target_requests:
            return []

        # 1. 将 Token 逐条写回对应的 Request 实例
        if next_tokens is not None:
            if isinstance(next_tokens, torch.Tensor):
                next_tokens_list = next_tokens.flatten().tolist()
            elif isinstance(next_tokens, list):
                next_tokens_list = next_tokens
            elif isinstance(next_tokens, dict):
                next_tokens_list = [next_tokens[req.request_id] for req in target_requests]
            else:
                raise TypeError(f"Unsupported next_tokens type: {type(next_tokens)}.")

            if len(next_tokens_list) != len(target_requests):
                raise ValueError(
                    f"Mismatch between number of tokens ({len(next_tokens_list)}) "
                    f"and number of requests ({len(target_requests)})."
                )

            for req, token in zip(target_requests, next_tokens_list):
                if not req.is_finished:
                    req.add_token(int(token))
                    if eos_token_id is not None and int(token) == eos_token_id:
                        req.mark_finished("stop")

        # 2. 检查并弹出已完成的请求，释放槽位
        just_finished: list[Request] = []
        for req in target_requests:
            if req.is_finished:
                if req.request_id in self.running_requests:
                    del self.running_requests[req.request_id]
                just_finished.append(req)

        return just_finished

    def abort_request(self, request_id: str) -> bool:
        """主动取消指定请求 (无论在排队中还是在运行中)。

        参数:
            request_id (str): 请求 ID。

        返回:
            bool: 是否成功找到并取消。
        """
        # 1. 检查运行中池
        if request_id in self.running_requests:
            req = self.running_requests.pop(request_id)
            req.mark_finished("aborted")
            return True

        # 2. 检查等待队列
        removed = self.request_queue.remove_request(request_id)
        if removed is not None:
            removed.mark_finished("aborted")
            return True

        return False

    def has_unfinished_requests(self) -> bool:
        """判断是否仍有未完成的请求 (运行中或排队中)。"""
        return len(self.running_requests) > 0 or not self.request_queue.is_empty()

    @property
    def num_running_requests(self) -> int:
        """当前处于运行状态的请求数量。"""
        return len(self.running_requests)

    @property
    def num_waiting_requests(self) -> int:
        """当前处于等待排队的请求数量。"""
        return len(self.request_queue)

    def __repr__(self) -> str:
        return (
            f"Scheduler(max_batch_size={self.max_batch_size}, "
            f"running={self.num_running_requests}, waiting={self.num_waiting_requests})"
        )
