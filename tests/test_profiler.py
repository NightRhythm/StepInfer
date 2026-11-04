"""TorchProfiler 单元测试。"""

import json
from pathlib import Path
import pytest
import torch

from stepinfer.utils.profiler import TorchProfiler


def test_torch_profiler_context_manager_lifecycle():
    profiler = TorchProfiler(enabled=True, use_cuda=False)
    assert not profiler.is_active

    with profiler:
        assert profiler.is_active
        x = torch.randn(32, 32)
        y = torch.matmul(x, x)

    # Context exit
    assert profiler.is_active
    table = profiler.summary(row_limit=5)
    assert isinstance(table, str)
    assert len(table) > 0


def test_torch_profiler_disabled_is_noop():
    profiler = TorchProfiler(enabled=False)
    with profiler:
        assert not profiler.is_active
        _ = torch.randn(10, 10) + 1.0

    assert not profiler.is_active
    assert "not enabled" in profiler.summary()


def test_torch_profiler_summary_and_sort():
    profiler = TorchProfiler(enabled=True, use_cuda=False)
    with profiler:
        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        _ = torch.mm(a, b)

    summary_cpu = profiler.summary(row_limit=10, sort_by="cpu_time_total")
    assert "mm" in summary_cpu or "aten::mm" in summary_cpu or len(summary_cpu) > 0


def test_torch_profiler_export_chrome_trace(tmp_path: Path):
    trace_file = tmp_path / "sub_dir" / "trace.json"
    profiler = TorchProfiler(enabled=True, use_cuda=False)

    with profiler:
        _ = torch.ones(10, 10) * 2.0

    profiler.export_chrome_trace(trace_file)
    assert trace_file.exists()
    assert trace_file.stat().st_size > 0

    # 验证是否为合法 JSON
    with open(trace_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, (dict, list))


def test_torch_profiler_export_before_run_raises():
    profiler = TorchProfiler()
    with pytest.raises(RuntimeError, match="Cannot export trace"):
        profiler.export_chrome_trace("dummy.json")
