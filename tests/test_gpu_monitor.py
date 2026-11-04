"""GPUMonitor 单元测试。"""

import pytest
import torch

from stepinfer.utils.gpu_monitor import GPUMemorySnapshot, GPUMonitor


def test_gpu_monitor_initialization_types():
    m_int = GPUMonitor(0)
    assert m_int.device.type == "cuda"

    m_str = GPUMonitor("cpu")
    assert m_str.device.type == "cpu"

    m_dev = GPUMonitor(torch.device("cpu"))
    assert m_dev.device.type == "cpu"

    with pytest.raises(TypeError, match="device must be int, str, or torch.device"):
        GPUMonitor(3.14)  # type: ignore


def test_gpu_monitor_cpu_fallback():
    monitor = GPUMonitor("cpu")
    assert not monitor.is_cuda
    assert monitor.get_device_name() == "CPU"

    # CPU 下 snapshot 返回全 0，不抛异常
    snapshot = monitor.snapshot()
    assert isinstance(snapshot, GPUMemorySnapshot)
    assert snapshot.allocated_mb == 0.0
    assert snapshot.max_allocated_mb == 0.0

    # CPU 下 reset_peak_stats 与 track_peak 不报错
    monitor.reset_peak_stats()
    with monitor.track_peak():
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_gpu_monitor_cuda_snapshot_and_track_peak():
    monitor = GPUMonitor("cuda:0")
    assert monitor.is_cuda
    assert monitor.get_device_name() != "CPU"

    # 测试显存分配与峰值检测
    with monitor.track_peak():
        # 分配大约 16MB 的张量 (4 * 1024 * 1024 * 4 bytes = 16MB)
        t = torch.empty((4096, 1024), dtype=torch.float32, device=monitor.device)
        snap = monitor.snapshot()
        assert snap.allocated_mb >= 16.0
        assert snap.max_allocated_mb >= 16.0
        assert snap.total_mb > 0.0

    # 释放显存后再次观察
    del t
    torch.cuda.empty_cache()
    snap_after = monitor.snapshot()
    assert snap_after.allocated_mb < snap.allocated_mb
