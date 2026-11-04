"""推理基准评测指标采集器，支持高精度 CUDA Event 计时、延迟分布与 MFU 统计。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
import torch

from stepinfer.utils.gpu_monitor import GPUMonitor


@dataclass(frozen=True)
class BenchmarkResult:
    """单次基准评测的结构化指标结果。"""

    prompt_tokens: int
    generated_tokens: int
    total_time_sec: float
    ttft_ms: float                     # 首 Token 延迟（Prefill 耗时，单位：ms）
    avg_itl_ms: float                  # 平均单步解码延迟（单位：ms）
    p50_itl_ms: float                  # P50 解码延迟（单位：ms）
    p90_itl_ms: float                  # P90 解码延迟（单位：ms）
    p99_itl_ms: float                  # P99 解码延迟（单位：ms）
    throughput_tok_per_sec: float      # 端到端总吞吐（tokens/s）
    peak_memory_mb: float              # 峰值显存（MB）
    mfu_percent: float | None = None   # Model FLOPs Utilization 百分比 (%)

    def summary(self) -> str:
        """返回多行易读的指标汇总文本。"""
        mfu_str = f"{self.mfu_percent:.2f}%" if self.mfu_percent is not None else "N/A"
        return (
            f"Prompt: {self.prompt_tokens} tokens | Generated: {self.generated_tokens} tokens\n"
            f"Total Time: {self.total_time_sec:.3f} s | Throughput: {self.throughput_tok_per_sec:.2f} tok/s\n"
            f"TTFT: {self.ttft_ms:.2f} ms | Avg ITL: {self.avg_itl_ms:.2f} ms "
            f"(P50: {self.p50_itl_ms:.2f} ms, P90: {self.p90_itl_ms:.2f} ms, P99: {self.p99_itl_ms:.2f} ms)\n"
            f"Peak VRAM: {self.peak_memory_mb:.2f} MB | MFU: {mfu_str}"
        )


class BenchmarkTracker:
    """基于 CUDA Event / 高精度计时器的端到端推理指标追踪器。"""

    # 常见 GPU 理论半精度（FP16/BF16）峰值 TFLOPS 参考表
    PEAK_TFLOPS_MAP = {
        "5060 ti": 45.0,  # RTX 5060 Ti
        "4090": 330.0,    # RTX 4090
        "3090": 142.0,    # RTX 3090
        "l40s": 362.0,    # L40S
        "a100": 312.0,    # A100 SXM 80GB
        "h100": 1979.0,   # H100 SXM
        "v100": 125.0,    # V100
    }

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        num_params: int = 0,
        peak_tflops: float | None = None,
    ) -> None:
        """初始化追踪器。

        参数：
            device: 目标设备。
            num_params: 模型参数量，用于估算 MFU。
            peak_tflops: 可选的手动指定 GPU 峰值 TFLOPS。若不指定则根据 GPU 型号自动推导。
        """
        self.device = torch.device(device)
        self.num_params = num_params
        self.custom_peak_tflops = peak_tflops
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.gpu_monitor = GPUMonitor(self.device)

        self._started = False
        self._cuda_events: List[torch.cuda.Event] = []
        self._cpu_times: List[float] = []

    def start(self) -> None:
        """启动追踪，记录起点并重置显存监控计数器。"""
        self._cuda_events.clear()
        self._cpu_times.clear()
        self.gpu_monitor.reset_peak_stats()

        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._cuda_events.append(event)
        else:
            self._cpu_times.append(time.perf_counter())

        self._started = True

    def record_step(self) -> None:
        """每完成一个生成步（Token）时记录打点。"""
        if not self._started:
            raise RuntimeError("Cannot record step: BenchmarkTracker has not been started. Call start() first.")

        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._cuda_events.append(event)
        else:
            self._cpu_times.append(time.perf_counter())

    def finish(self, prompt_tokens: int) -> BenchmarkResult:
        """结束追踪并结算全量延迟、吞吐与显存指标。

        参数：
            prompt_tokens: 输入 Prompt 的 Token 数量。
        """
        if not self._started:
            raise RuntimeError("Cannot finish: BenchmarkTracker has not been started.")

        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")

        # 1. 计算每个生成步的时间差（毫秒 ms）
        deltas_ms: List[float] = []

        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            for i in range(len(self._cuda_events) - 1):
                ms = self._cuda_events[i].elapsed_time(self._cuda_events[i + 1])
                deltas_ms.append(ms)
        else:
            for i in range(len(self._cpu_times) - 1):
                ms = (self._cpu_times[i + 1] - self._cpu_times[i]) * 1000.0
                deltas_ms.append(ms)

        if not deltas_ms:
            raise RuntimeError("No generation steps recorded. Did you call record_step()?")

        # 2. TTFT 与 ITL 切分
        ttft_ms = deltas_ms[0]
        itl_ms = deltas_ms[1:] if len(deltas_ms) > 1 else [deltas_ms[0]]

        total_time_sec = sum(deltas_ms) / 1000.0
        generated_tokens = len(deltas_ms)
        throughput = generated_tokens / total_time_sec if total_time_sec > 0 else 0.0

        # 3. 分位线与均值
        sorted_itl = sorted(itl_ms)
        avg_itl = sum(sorted_itl) / len(sorted_itl)
        p50 = sorted_itl[int(len(sorted_itl) * 0.50)]
        p90 = sorted_itl[int(len(sorted_itl) * 0.90)]
        p99 = sorted_itl[int(len(sorted_itl) * 0.99)]

        # 4. 显存峰值
        snapshot = self.gpu_monitor.snapshot()

        # 5. MFU 计算
        mfu = self._compute_mfu(throughput)

        self._started = False

        return BenchmarkResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_time_sec=round(total_time_sec, 4),
            ttft_ms=round(ttft_ms, 2),
            avg_itl_ms=round(avg_itl, 2),
            p50_itl_ms=round(p50, 2),
            p90_itl_ms=round(p90, 2),
            p99_itl_ms=round(p99, 2),
            throughput_tok_per_sec=round(throughput, 2),
            peak_memory_mb=snapshot.max_allocated_mb,
            mfu_percent=round(mfu, 4) if mfu is not None else None,
        )

    def _compute_mfu(self, throughput: float) -> float | None:
        """估算 MFU (Model FLOPs Utilization)。"""
        if self.num_params <= 0:
            return None

        peak_tflops = self.custom_peak_tflops
        if peak_tflops is None:
            if not self.use_cuda:
                return None
            dev_name = torch.cuda.get_device_name(self.device).lower()
            for key, tflops in self.PEAK_TFLOPS_MAP.items():
                if key in dev_name:
                    peak_tflops = tflops
                    break

        if peak_tflops is None or peak_tflops <= 0:
            return None

        # 传统 Transformer 前向运算量估算：每个 token 约进行 2 * num_params 次浮点运算
        actual_flops = 2 * self.num_params * throughput
        return (actual_flops / (peak_tflops * 1e12)) * 100.0
