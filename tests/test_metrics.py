"""BenchmarkTracker 与 BenchmarkResult 单元测试。"""

import time
import pytest
import torch

from benchmarks.metrics import BenchmarkResult, BenchmarkTracker, aggregate_results


def test_benchmark_tracker_cpu_lifecycle():
    tracker = BenchmarkTracker(device="cpu")
    tracker.start()

    # 模拟生成 3 个 token
    for _ in range(3):
        time.sleep(0.01)
        tracker.record_step()

    result = tracker.finish(prompt_tokens=10)
    assert isinstance(result, BenchmarkResult)
    assert result.prompt_tokens == 10
    assert result.generated_tokens == 3
    assert result.generation_wall_time_sec > 0.02
    assert result.first_token_interval_ms > 0.0
    assert result.avg_itl_ms is not None and result.avg_itl_ms > 0.0
    assert result.p50_itl_ms is not None and result.p50_itl_ms > 0.0
    assert result.p90_itl_ms is not None and result.p90_itl_ms > 0.0
    assert result.p99_itl_ms is not None and result.p99_itl_ms > 0.0
    assert result.generation_wall_throughput_tok_per_sec > 0.0
    assert result.timing_backend == "cpu_perf_counter"
    assert len(result.step_intervals_ms) == 3

    # 验证字典转换与文本摘要
    data_dict = result.to_dict()
    assert isinstance(data_dict, dict)
    assert data_dict["prompt_tokens"] == 10

    summary_text = result.summary()
    assert "First-token interval:" in summary_text
    assert "Avg ITL:" in summary_text
    assert "Wall throughput:" in summary_text


def test_benchmark_tracker_single_step():
    tracker = BenchmarkTracker(device="cpu")
    tracker.start()
    time.sleep(0.005)
    tracker.record_step()

    result = tracker.finish(prompt_tokens=5)
    assert result.generated_tokens == 1
    assert result.first_token_interval_ms > 0.0
    # 单 token 生成时无后续解码步，ITL 为 None
    assert result.avg_itl_ms is None
    assert result.p50_itl_ms is None


def test_benchmark_tracker_validation_errors():
    tracker = BenchmarkTracker(device="cpu")

    # 未调用 start 直接 record_step
    with pytest.raises(RuntimeError, match="has not been started"):
        tracker.record_step()

    # 未调用 start 直接 finish
    with pytest.raises(RuntimeError, match="has not been started"):
        tracker.finish(prompt_tokens=10)

    # 重复调用 start
    tracker.start()
    with pytest.raises(RuntimeError, match="already started"):
        tracker.start()

    # 启动了但没记录任何步
    with pytest.raises(RuntimeError, match="No generation steps recorded"):
        tracker.finish(prompt_tokens=10)

    # 非法 prompt_tokens
    tracker.start()
    tracker.record_step()
    with pytest.raises(ValueError, match="prompt_tokens must be a positive integer"):
        tracker.finish(prompt_tokens=0)

    # 不支持的设备类型
    with pytest.raises(ValueError, match="Only CPU and CUDA"):
        BenchmarkTracker(device="meta")


def test_aggregate_results():
    tracker1 = BenchmarkTracker(device="cpu")
    tracker1.start()
    time.sleep(0.005)
    tracker1.record_step()
    tracker1.record_step()
    res1 = tracker1.finish(prompt_tokens=10)

    tracker2 = BenchmarkTracker(device="cpu")
    tracker2.start()
    time.sleep(0.005)
    tracker2.record_step()
    tracker2.record_step()
    res2 = tracker2.finish(prompt_tokens=10)

    summary = aggregate_results([res1, res2])
    assert "generation_wall_time_sec" in summary
    assert "first_token_interval_ms" in summary
    assert "median" in summary["generation_wall_time_sec"]
    assert "min" in summary["generation_wall_time_sec"]
    assert "max" in summary["generation_wall_time_sec"]

    # 空列表应抛出异常
    with pytest.raises(ValueError, match="cannot be empty"):
        aggregate_results([])


def test_benchmark_tracker_mfu_calculation():
    # 模拟已知 peak_tflops 和 num_params 的 MFU 计算
    tracker = BenchmarkTracker(
        device="cpu",
        num_params=500_000_000,
        peak_tflops=10.0,
    )
    tracker.start()
    time.sleep(0.01)
    tracker.record_step()
    res = tracker.finish(prompt_tokens=10)

    assert res.mfu_percent is not None
    assert res.mfu_percent > 0.0
    assert "MFU:" in res.summary()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test")
def test_benchmark_tracker_cuda_lifecycle():
    tracker = BenchmarkTracker(device="cuda:0")
    tracker.start()

    # 模拟 3 次 GPU 前向计算
    for _ in range(3):
        x = torch.randn(512, 512, device="cuda:0")
        _ = torch.matmul(x, x)
        tracker.record_step()

    res = tracker.finish(prompt_tokens=32)
    assert res.generated_tokens == 3
    assert res.first_token_interval_ms > 0.0
    assert res.peak_allocated_mb >= 0.0
    assert res.generation_wall_throughput_tok_per_sec > 0.0
    assert res.timing_backend == "cuda_event"
    assert len(res.step_intervals_ms) == 3
