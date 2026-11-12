"""受控单请求生成运行的计时和显存指标。"""

from __future__ import annotations

import statistics
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from stepinfer.utils.gpu_monitor import GPUMonitor

GPU_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "rtx 5060 ti": {"fp32": 22.5, "bf16": 45.0, "fp16": 45.0},
    "rtx 4090": {"fp32": 82.6, "tf32": 165.0, "fp16": 330.0, "bf16": 330.0},
    "rtx 3090": {"fp32": 35.6, "tf32": 71.0, "fp16": 142.0, "bf16": 142.0},
    "a100": {"fp32": 19.5, "tf32": 156.0, "fp16": 312.0, "bf16": 312.0},
    "h100": {"fp32": 67.0, "tf32": 990.0, "fp16": 1979.0, "bf16": 1979.0},
    "h200": {"fp32": 67.0, "tf32": 990.0, "fp16": 1979.0, "bf16": 1979.0},
    "l40s": {"fp32": 91.6, "tf32": 183.0, "fp16": 362.0, "bf16": 362.0},
}


def get_peak_tflops(device: torch.device | str, dtype: torch.dtype) -> float | None:
    """根据 GPU 型号与数据类型返回理论密集 TFLOPS 峰值。"""
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        return None

    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    dtype_key = dtype_map.get(dtype)
    if dtype_key is None:
        return None

    device_name = torch.cuda.get_device_name(target_device).lower()
    for model_key, tflops_table in GPU_PEAK_TFLOPS.items():
        if model_key in device_name:
            return tflops_table.get(dtype_key)
    return None


@dataclass(frozen=True)
class BenchmarkResult:
    """一次测量运行的结果；设备计时不包含分词和文本解码。"""

    prompt_tokens: int
    generated_tokens: int
    generation_wall_time_sec: float
    first_token_interval_ms: float
    timing_backend: str
    step_intervals_ms: tuple[float, ...]
    avg_itl_ms: float | None
    p50_itl_ms: float | None
    p90_itl_ms: float | None
    p99_itl_ms: float | None
    generation_wall_throughput_tok_per_sec: float
    allocated_before_mb: float
    reserved_before_mb: float
    peak_allocated_mb: float
    peak_allocated_delta_mb: float
    peak_reserved_mb: float
    peak_reserved_delta_mb: float
    device_free_after_mb: float
    mfu_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        def fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.2f}"

        mfu_str = f" | MFU: {self.mfu_percent:.2f}%" if self.mfu_percent is not None else ""
        return (
            f"Prompt: {self.prompt_tokens} tokens | Generated: {self.generated_tokens} tokens\n"
            f"Generation wall time: {self.generation_wall_time_sec:.3f} s | "
            f"Wall throughput: {self.generation_wall_throughput_tok_per_sec:.2f} tok/s{mfu_str}\n"
            f"First-token interval: {self.first_token_interval_ms:.2f} ms | "
            f"Avg ITL: {fmt(self.avg_itl_ms)} ms "
            f"(P50: {fmt(self.p50_itl_ms)} ms, "
            f"P90: {fmt(self.p90_itl_ms)} ms, "
            f"P99: {fmt(self.p99_itl_ms)} ms)\n"
            f"Peak allocated: {self.peak_allocated_mb:.2f} MiB "
            f"(+{self.peak_allocated_delta_mb:.2f} MiB over baseline) | "
            f"Peak reserved: {self.peak_reserved_mb:.2f} MiB "
            f"(+{self.peak_reserved_delta_mb:.2f} MiB over baseline)"
        )


class BenchmarkTracker:
    """测量模型生成墙钟耗时和逐 token 的设备时间间隔。"""

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        *,
        num_params: int = 0,
        dtype: torch.dtype = torch.float32,
        peak_tflops: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA timing are supported.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.gpu_monitor = GPUMonitor(self.device)
        self.num_params = num_params
        self.dtype = dtype
        self.peak_tflops = peak_tflops if peak_tflops is not None else get_peak_tflops(self.device, self.dtype)
        self._started = False
        self._wall_start = 0.0
        self._wall_end: float | None = None
        self._stream: torch.cuda.Stream | None = None
        self._cuda_events: list[torch.cuda.Event] = []
        self._cpu_step_times: list[float] = []
        self._allocated_before_mb = 0.0
        self._reserved_before_mb = 0.0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("BenchmarkTracker is already started.")

        self._cuda_events.clear()
        self._cpu_step_times.clear()
        self._wall_end = None
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        self.gpu_monitor.reset_peak_stats()

        baseline = self.gpu_monitor.snapshot()
        self._allocated_before_mb = baseline.allocated_mb
        self._reserved_before_mb = baseline.reserved_mb

        if self.use_cuda:
            self._stream = torch.cuda.current_stream(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            self._wall_start = time.perf_counter()
            start_event.record(self._stream)
            self._cuda_events.append(start_event)
        else:
            self._wall_start = time.perf_counter()
        self._started = True

    def record_step(self) -> None:
        """在当前设备流上记录一个已经完成的生成 token 步骤。"""
        if not self._started:
            raise RuntimeError("Cannot record step: BenchmarkTracker has not been started. Call start() first.")

        if self.use_cuda:
            assert self._stream is not None
            event = torch.cuda.Event(enable_timing=True)
            event.record(self._stream)
            self._cuda_events.append(event)
        else:
            self._cpu_step_times.append(time.perf_counter())

    def finish(self, prompt_tokens: int, batch_size: int = 1) -> BenchmarkResult:
        """同步设备、计算指标并结束当前测量运行。

        参数:
            prompt_tokens (int): 序列的 Prompt Token 数量。
            batch_size (int): 批处理大小 (默认为 1)。
        """
        if not self._started:
            raise RuntimeError("Cannot finish: BenchmarkTracker has not been started.")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        try:
            if self.use_cuda:
                torch.cuda.synchronize(self.device)
            self._wall_end = time.perf_counter()

            if self.use_cuda:
                intervals = [
                    self._cuda_events[index].elapsed_time(self._cuda_events[index + 1])
                    for index in range(len(self._cuda_events) - 1)
                ]
            else:
                intervals = [
                    (self._cpu_step_times[0] - self._wall_start) * 1000.0,
                    *[
                        (self._cpu_step_times[index] - self._cpu_step_times[index - 1]) * 1000.0
                        for index in range(1, len(self._cpu_step_times))
                    ],
                ] if self._cpu_step_times else []

            if not intervals:
                raise RuntimeError("No generation steps recorded. Did you call record_step()?")

            first_token_ms = intervals[0]
            itl = intervals[1:]
            avg_itl = sum(itl) / len(itl) if itl else None

            def percentile(values: list[float], quantile: float) -> float | None:
                if not values:
                    return None
                ordered = sorted(values)
                # 在 (n - 1) * q 位置做线性插值，同时兼容 n == 1。
                position = quantile * (len(ordered) - 1)
                lower = math.floor(position)
                upper = math.ceil(position)
                return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

            snapshot = self.gpu_monitor.snapshot()
            wall_time = max(0.0, self._wall_end - self._wall_start)
            generated_tokens = len(intervals) * batch_size
            throughput = generated_tokens / wall_time if wall_time > 0 else 0.0

            mfu_percent: float | None = None
            if (
                self.peak_tflops is not None
                and self.peak_tflops > 0
                and self.num_params > 0
                and throughput > 0
            ):
                mfu_percent = round(
                    (2.0 * self.num_params * throughput) / (self.peak_tflops * 1e12) * 100.0,
                    2,
                )

            return BenchmarkResult(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                generation_wall_time_sec=wall_time,
                first_token_interval_ms=first_token_ms,
                timing_backend="cuda_event" if self.use_cuda else "cpu_perf_counter",
                step_intervals_ms=tuple(intervals),
                avg_itl_ms=avg_itl,
                p50_itl_ms=percentile(itl, 0.50),
                p90_itl_ms=percentile(itl, 0.90),
                p99_itl_ms=percentile(itl, 0.99),
                generation_wall_throughput_tok_per_sec=throughput,
                allocated_before_mb=self._allocated_before_mb,
                reserved_before_mb=self._reserved_before_mb,
                peak_allocated_mb=snapshot.max_allocated_mb,
                peak_allocated_delta_mb=round(
                    max(0.0, snapshot.max_allocated_mb - self._allocated_before_mb), 2
                ),
                peak_reserved_mb=snapshot.max_reserved_mb,
                peak_reserved_delta_mb=round(
                    max(0.0, snapshot.max_reserved_mb - self._reserved_before_mb), 2
                ),
                device_free_after_mb=snapshot.free_mb,
                mfu_percent=mfu_percent,
            )
        finally:
            self._started = False
            self._stream = None


def aggregate_results(results: list[BenchmarkResult]) -> dict[str, dict[str, float]]:
    """汇总多次运行的中位数以及观测到的最小/最大值。"""
    if not results:
        raise ValueError("results cannot be empty")

    metrics = (
        "generation_wall_time_sec",
        "first_token_interval_ms",
        "avg_itl_ms",
        "p50_itl_ms",
        "p90_itl_ms",
        "p99_itl_ms",
        "generation_wall_throughput_tok_per_sec",
        "mfu_percent",
        "peak_allocated_mb",
        "peak_allocated_delta_mb",
        "peak_reserved_mb",
        "peak_reserved_delta_mb",
    )
    summary: dict[str, dict[str, float]] = {}
    for name in metrics:
        values = [getattr(result, name) for result in results]
        present = [float(value) for value in values if value is not None]
        if present:
            summary[name] = {
                "median": round(statistics.median(present), 6),
                "min": round(min(present), 6),
                "max": round(max(present), 6),
            }
    return summary
