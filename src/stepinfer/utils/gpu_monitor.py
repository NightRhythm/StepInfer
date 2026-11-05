"""汇总 PyTorch CUDA 分配器用量和设备级剩余显存。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch


_MIB = 1024 ** 2


@dataclass(frozen=True)
class GPUMemorySnapshot:
    """显存单位为 MiB；峰值覆盖最近一次重置统计后的区间。"""

    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    total_mb: float
    free_mb: float


class GPUMonitor:
    """观察单个 CUDA 设备，不主动改变分配器状态。"""

    def __init__(self, device: str | torch.device | int = 0) -> None:
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, (str, torch.device)):
            self.device = torch.device(device)
        else:
            raise TypeError(f"device must be int, str, or torch.device, got {type(device)}")

        self._is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if self._is_cuda and self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.last_peak_snapshot: GPUMemorySnapshot | None = None

    @property
    def is_cuda(self) -> bool:
        return self._is_cuda

    def get_device_name(self) -> str:
        if not self._is_cuda:
            return "CPU"
        return torch.cuda.get_device_name(self.device)

    def reset_peak_stats(self) -> None:
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> GPUMemorySnapshot:
        """返回分配器用量，以及设备级的剩余/总显存。"""
        if not self._is_cuda:
            return GPUMemorySnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        free, total = torch.cuda.mem_get_info(self.device)
        return GPUMemorySnapshot(
            allocated_mb=torch.cuda.memory_allocated(self.device) / _MIB,
            reserved_mb=torch.cuda.memory_reserved(self.device) / _MIB,
            max_allocated_mb=torch.cuda.max_memory_allocated(self.device) / _MIB,
            max_reserved_mb=torch.cuda.max_memory_reserved(self.device) / _MIB,
            total_mb=total / _MIB,
            free_mb=free / _MIB,
        )

    @contextmanager
    def track_peak(self) -> Generator[GPUMonitor, None, None]:
        """进入时同步并重置统计，退出时将快照保存到 last_peak_snapshot。

        峰值计数器是当前进程和设备级的，不要嵌套使用或并发重置。
        CPU 模式下 CUDA 统计字段均为 0。
        """
        self.last_peak_snapshot = None
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
        self.reset_peak_stats()
        try:
            yield self
        finally:
            if self._is_cuda:
                torch.cuda.synchronize(self.device)
            self.last_peak_snapshot = self.snapshot()
