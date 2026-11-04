"""PyTorch Profiler 包装器，支持算子耗时剖析与 Chrome Trace 导出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.profiler


class TorchProfiler:
    """包装 torch.profiler.profile 的上下文分析器（对齐 vLLM / 现代推理系统风格）。"""

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
            record_shapes: 是否记录每个算子的输入/输出 Tensor 维度。
            profile_memory: 是否追踪张量内存分配/释放事件。
            with_stack: 是否记录 Python 调用栈（会略微增加开销）。
        """
        self.enabled = enabled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self._prof: torch.profiler.profile | None = None

    @property
    def is_active(self) -> bool:
        """Profiler 是否处于活跃运行状态。"""
        return self._prof is not None

    def __enter__(self) -> TorchProfiler:
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
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._prof is not None:
            self._prof.__exit__(exc_type, exc_val, exc_tb)

    def summary(self, row_limit: int = 15, sort_by: str | None = None) -> str:
        """生成格式化的算子耗时控制台表格。

        参数：
            row_limit: 展示耗时最高的前 N 个算子。
            sort_by: 排序指标，默认 GPU 可用时按 'cuda_time_total'，否则按 'cpu_time_total'。
        """
        if self._prof is None:
            return "Profiler was not enabled or executed."

        if sort_by is None:
            sort_by = "cuda_time_total" if self.use_cuda else "cpu_time_total"

        return self._prof.key_averages().table(
            sort_by=sort_by,
            row_limit=row_limit,
        )

    def export_chrome_trace(self, output_path: str | Path) -> None:
        """导出 Chrome Trace JSON 文件，可在 chrome://tracing 或 perfetto.dev 可视化查看。"""
        if self._prof is None:
            raise RuntimeError("Cannot export trace: Profiler has not run yet.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._prof.export_chrome_trace(str(out_path))
