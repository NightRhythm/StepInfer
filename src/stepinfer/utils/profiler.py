"""PyTorch Profiler 包装器，支持算子耗时剖析与 Chrome Trace 导出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.profiler


class TorchProfiler:
    """包装 torch.profiler.profile 的上下文分析器。"""

    def __init__(
        self,
        enabled: bool = True,
        use_cuda: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> None:
        """初始化 Profiler 配置。

        参数：
            enabled: 是否启用 Profiling（关闭时作为空上下文，避免产生额外性能开销）。
            use_cuda: 是否记录 CUDA Kernel 活动。
            record_shapes: 是否记录算子输入 Tensor 维度（可能增加开销与引用保留）。
            profile_memory: 是否追踪张量内存分配/释放事件。
            with_stack: 是否记录 Python 调用栈（会增加采集开销）。
        """
        self.enabled = enabled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self._prof: torch.profiler.profile | None = None
        self._active = False
        self._completed = False

    @property
    def is_active(self) -> bool:
        """当前实例是否已经创建过性能分析会话。

        退出上下文后该属性仍为 True，便于调用方继续请求汇总或导出
        时间线；如果需要判断当前是否仍在上下文中采集，请使用
        :attr:`is_running`。
        """
        return self._prof is not None

    @property
    def is_running(self) -> bool:
        """当前是否正在上下文中采集性能数据。"""
        return self._active

    def __enter__(self) -> TorchProfiler:
        if self._active:
            raise RuntimeError("TorchProfiler cannot be entered recursively.")
        self._prof = None
        self._completed = False
        if not self.enabled:
            return self

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._prof = torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._prof.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._prof is not None and self._active:
            try:
                self._prof.__exit__(exc_type, exc_val, exc_tb)
                self._completed = True
            finally:
                self._active = False

    def summary(self, row_limit: int = 15, sort_by: str | None = None) -> str:
        """生成格式化的算子耗时控制台表格。

        参数：
            row_limit: 展示耗时最高的前 N 个算子。
            sort_by: 默认按算子自身的 CUDA/CPU 耗时排序，避免父子算子
                的累计耗时重复归因。
        """
        if self._prof is None:
            return "Profiler was not enabled or executed."
        if not self._completed:
            raise RuntimeError("Exit the profiler context before requesting a summary.")

        if sort_by is None:
            sort_by = "self_cuda_time_total" if self.use_cuda else "self_cpu_time_total"

        return self._prof.key_averages().table(
            sort_by=sort_by,
            row_limit=row_limit,
        )

    def export_chrome_trace(self, output_path: str | Path) -> None:
        """导出 Chrome Trace JSON 文件，可用 Chrome tracing 或 Perfetto 查看。"""
        if self._prof is None or not self._completed:
            raise RuntimeError("Cannot export trace: Profiler has not run yet.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._prof.export_chrome_trace(str(out_path))
