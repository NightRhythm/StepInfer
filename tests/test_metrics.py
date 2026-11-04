"""BenchmarkTracker 与 BenchmarkResult 单元测试。"""

import time
import pytest
import torch

from benchmarks.metrics import BenchmarkResult, BenchmarkTracker


def test_benchmark_tracker_cpu_lifecycle():
    tracker = BenchmarkTracker(device="cpu", num_params=100_000_000, peak_tflops=1.0)
    tracker.start()

    # 模拟生成 3 个 token
    for _ in range(3):
        time.sleep(0.01)
        tracker.record_step()

    result = tracker.finish(prompt_tokens=10)
    assert isinstance(result, BenchmarkResult)
    assert result.prompt_tokens == 10
    assert result.generated_tokens == 3
    assert result.total_time_sec > 0.02
    assert result.ttft_ms > 0.0
    assert result.avg_itl_ms > 0.0
    assert result.throughput_tok_per_sec > 0.0
    assert result.mfu_percent is not None
    assert result.mfu_percent > 0.0

    summary_text = result.summary()
    assert "TTFT:" in summary_text
    assert "Avg ITL:" in summary_text
    assert "Throughput:" in summary_text


def test_benchmark_tracker_single_step():
    tracker = BenchmarkTracker(device="cpu")
    tracker.start()
    time.sleep(0.005)
    tracker.record_step()

    result = tracker.finish(prompt_tokens=5)
    assert result.generated_tokens == 1
    assert result.ttft_ms > 0.0
    assert result.avg_itl_ms == result.ttft_ms


def test_benchmark_tracker_validation_errors():
    tracker = BenchmarkTracker(device="cpu")

    # 未调用 start 直接 record_step
    with pytest.raises(RuntimeError, match="has not been started"):
        tracker.record_step()

    # 未调用 start 直接 finish
    with pytest.raises(RuntimeError, match="has not been started"):
        tracker.finish(prompt_tokens=10)

    # 启动了但没记录任何步
    tracker.start()
    with pytest.raises(RuntimeError, match="No generation steps recorded"):
        tracker.finish(prompt_tokens=10)

    # 非法 prompt_tokens
    tracker.start()
    tracker.record_step()
    with pytest.raises(ValueError, match="prompt_tokens must be a positive integer"):
        tracker.finish(prompt_tokens=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_benchmark_tracker_cuda_lifecycle():
    tracker = BenchmarkTracker(device="cuda:0", num_params=600_000_000)
    tracker.start()

    # 模拟 3 次 GPU 前向计算
    for _ in range(3):
        x = torch.randn(512, 512, device="cuda:0")
        _ = torch.matmul(x, x)
        tracker.record_step()

    res = tracker.finish(prompt_tokens=32)
    assert res.generated_tokens == 3
    assert res.ttft_ms > 0.0
    assert res.peak_memory_mb >= 0.0
    assert res.throughput_tok_per_sec > 0.0
