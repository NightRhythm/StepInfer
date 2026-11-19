# Throughput Benchmark — Qwen3-0.6B-Base

## Setup

| Parameter | Value |
|:---|:---|
| Model | Qwen3-0.6B-Base |
| Parameters | 596,049,920 |
| Precision | bf16 |
| Device | NVIDIA GeForce RTX 5060 Ti |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.16 |
| Scheduler | Continuous Batching vs Static Baseline |
| Concurrency (Slots) | 4 |
| Total Requests | 8 |
| Sampling | Greedy (argmax) |

---

## Results

### 1. Aggregate Summary (汇总统计对比)

| Metric | Static Batching | Continuous Batching | Delta |
|:---|---:|---:|---:|
| Total Wall Time (s) | 2.197 | 1.921 | 1.14x speedup |
| Throughput (tok/s) | 87.83 | 109.86 | +25.1% |
| Avg Latency (s) | 1.491 | 1.081 | -27.5% |
| P50 Latency (s) | 0.786 | 0.621 | -21.0% |
| P90 Latency (s) | 2.197 | 1.921 | -12.6% |
| Min Latency (s) | 0.786 | 0.308 | 2.55x |
| Max Latency (s) | 2.197 | 1.921 | 1.14x |
| Total Prompt Tokens | 80 | 80 | - |
| Total Output Tokens | 193 | 211 | +9.3% |
| Peak Memory (MB) | 2273.1 | 2271.4 | -1.7 MB |

### 2. Per-Request Breakdown (逐请求原始测量明细)

| Req ID | Prompt Len | Target Tok | Output Tok (Static) | Output Tok (Cont) | Static Latency (s) | Continuous Latency (s) | Speedup | Latency Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 10 | 10 | 0.786 | 0.394 | 2.00x | -49.9% |
| 1 | 6 | 8 | 8 | 8 | 0.786 | 0.308 | 2.55x | -60.8% |
| 2 | 7 | 16 | 16 | 16 | 0.786 | 0.621 | 1.27x | -21.0% |
| 3 | 10 | 15 | 15 | 15 | 0.786 | 0.560 | 1.40x | -28.7% |
| 4 | 12 | 40 | 40 | 40 | 2.197 | 1.558 | 1.41x | -29.1% |
| 5 | 14 | 34 | 16 | 34 | 2.197 | 1.443 | 1.52x | -34.3% |
| 6 | 12 | 43 | 43 | 43 | 2.197 | 1.848 | 1.19x | -15.9% |
| 7 | 9 | 45 | 45 | 45 | 2.197 | 1.921 | 1.14x | -12.6% |
