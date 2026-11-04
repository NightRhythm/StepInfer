"""轻量级 GPU 显存监控工具，用于追踪张量显存分配与推理过程中的显存峰值。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator
import torch


@dataclass(frozen=True)
class GPUMemorySnapshot:
    """GPU 显存快照（单位：MB）。"""

    allocated_mb: float       # 当前 PyTorch 张量实际占用显存
    reserved_mb: float        # PyTorch 显存缓存池（Caching Allocator）占用显存
    max_allocated_mb: float   # 统计区间内 PyTorch 张量显存峰值
    total_mb: float           # 设备物理总显存
    free_mb: float            # 物理剩余显存 (total - reserved)


class GPUMonitor:
    """基于原生 PyTorch 的 GPU 显存监控器。"""

    def __init__(self, device: str | torch.device | int = 0) -> None:
        """初始化显存监控器。

        参数：
            device: 设备标号或名称（如 0, "cuda:0", "cpu", torch.device("cuda:0")）。
        """
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"device must be int, str, or torch.device, got {type(device)}")

        self._is_cuda = self.device.type == "cuda" and torch.cuda.is_available()

    @property
    def is_cuda(self) -> bool:
        """当前监控设备是否为可用 CUDA 设备。"""
        return self._is_cuda

    def get_device_name(self) -> str:
        """获取设备名称。"""
        if not self._is_cuda:
            return "CPU"
        device_idx = self.device.index or 0
        return torch.cuda.get_device_name(device_idx)

    def reset_peak_stats(self) -> None:
        """重置显存峰值统计计数器。"""
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> GPUMemorySnapshot:
        """采集当前时刻的显存状态快照。"""
        if not self._is_cuda:
            return GPUMemorySnapshot(0.0, 0.0, 0.0, 0.0, 0.0)

        device_idx = self.device.index or 0
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
        free = max(0.0, total - reserved)

        return GPUMemorySnapshot(
            allocated_mb=round(allocated, 2),
            reserved_mb=round(reserved, 2),
            max_allocated_mb=round(max_allocated, 2),
            total_mb=round(total, 2),
            free_mb=round(free, 2),
        )

    @contextmanager
    def track_peak(self) -> Generator[GPUMonitor, None, None]:
        """上下文管理器：自动在进入时重置峰值，并在退出时保留该区间的最大显存。"""
        self.reset_peak_stats()
        try:
            yield self
        finally:
            pass
