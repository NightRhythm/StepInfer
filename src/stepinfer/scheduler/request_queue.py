"""请求等待队列管理模块。

基于双端队列 (deque) 提供高效的 FIFO (先进先出) 组织机制与容量满载背压防护。
"""

from collections import deque

from stepinfer.scheduler.request import Request, RequestStatus


class QueueFullError(Exception):
    """请求队列满载异常。触发背压机制 (Backpressure)。"""
    pass


class RequestQueue:
    """就绪与等待请求管理队列。

    专注于管理处于 WAITING 状态的候选请求，提供 $O(1)$ 的快速入队、批量出队与满载拦截。
    """

    def __init__(self, max_queue_size: int | None = None) -> None:
        """初始化请求队列。"""
        if max_queue_size is not None and max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be a positive integer, got {max_queue_size}.")

        self.max_queue_size = max_queue_size
        self._waiting_queue: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        """将新请求加入等待队列尾部 (FIFO)。"""
        if self.max_queue_size is not None and len(self._waiting_queue) >= self.max_queue_size:
            raise QueueFullError(
                f"Request queue is full (capacity={self.max_queue_size})."
            )

        if request.is_finished:
            raise ValueError(f"Cannot enqueue an already finished request ({request.request_id}).")

        # 确保标记为 WAITING
        request.status = RequestStatus.WAITING
        self._waiting_queue.append(request)

    def get_waiting_requests(self, max_requests: int | None = None) -> list[Request]:
        """按 FIFO 顺序从队头弹出指定数量的候选请求。"""
        if max_requests is not None and max_requests <= 0:
            raise ValueError(f"max_requests must be a positive integer, got {max_requests}.")

        target_count = len(self._waiting_queue) if max_requests is None else min(max_requests, len(self._waiting_queue))
        return [self._waiting_queue.popleft() for _ in range(target_count)]

    def peek_waiting_requests(self, max_requests: int | None = None) -> list[Request]:
        """查看队头的候选请求，但不将其移出队列 (只读透视)。"""
        if max_requests is not None:
            if max_requests <= 0:
                raise ValueError(f"max_requests must be a positive integer, got {max_requests}.")
            return list(self._waiting_queue)[:max_requests]
        return list(self._waiting_queue)

    def remove_request(self, request_id: str) -> Request | None:
        """从队列中检索并移除指定的等待请求 (用于客户端取消请求场景)。"""
        for req in self._waiting_queue:
            if req.request_id == request_id:
                self._waiting_queue.remove(req)
                return req
        return None

    def is_empty(self) -> bool:
        """判断等待队列是否为空。"""
        return len(self._waiting_queue) == 0

    def __len__(self) -> int:
        """返回当前处于等待状态的请求总数。"""
        return len(self._waiting_queue)

    def __repr__(self) -> str:
        return f"RequestQueue(size={len(self)}, max_size={self.max_queue_size})"
