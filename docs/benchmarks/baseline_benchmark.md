# Baseline Benchmark — Day 9 (No KV Cache, Single Request)

本文档记录 StepInfer 在无 KV Cache、单请求贪心解码设置下的基线性能基准测试。测试数据全部基于真实 GPU 运行生成，原始数据保存在 `docs/benchmarks/raw/` 目录下。

---

## Setup

| Parameter                | Qwen3-0.6B-Base                        | Qwen3-1.7B-Base                        |
|:------------------------ |:-------------------------------------- |:-------------------------------------- |
| Architecture             | Qwen3 (custom implementation)          | Qwen3 (custom implementation)          |
| Parameters               | 596,049,920 (227 state_dict keys)      | 1,720,574,976 (227 state_dict keys)    |
| Precision                | `bf16`                                 | `bf16`                                 |
| GPU                      | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| Architecture Generation  | Blackwell (SM 12.0)                    | Blackwell (SM 12.0)                    |
| Peak TFLOPS (dense bf16) | 45.0                                   | 45.0                                   |
| PyTorch                  | 2.9.1+cu128                            | 2.9.1+cu128                            |
| Python                   | 3.12.16                                | 3.12.16                                |
| KV Cache                 | None (Full-sequence recompute)         | None (Full-sequence recompute)         |
| Batching                 | Single request (batch_size=1)          | Single request (batch_size=1)          |
| Sampling                 | Greedy (argmax)                        | Greedy (argmax)                        |
| Timing Backend           | CUDA Event (`torch.cuda.Event`)        | CUDA Event (`torch.cuda.Event`)        |

---

## 1. Latency & Throughput Benchmark

**测试目标：** 测量不同 Prompt 长度对单请求端到端生成延迟、吞吐量、显存峰值及 MFU（模型算力利用率）的影响。

**测试配置：**

- 每轮固定生成 **32 个 token**（禁用 EOS 提前停止以保持生成步数完全一致）。
- 5 组 Prompt 长度：`[32, 64, 256, 512, 1024]`。
- 每个长度先执行 **3 轮完整预热**，再连续记录 **10 轮正式评测**。
- 表格数据汇报格式为：`中位数 [最小值, 最大值]`。

### Qwen3-0.6B-Base 评测结果

原始数据：[`raw/md/baseline_latency_0.6b.md`](raw/md/baseline_latency_0.6b.md) | [`raw/json/baseline_latency_0.6b.json`](raw/json/baseline_latency_0.6b.json)

| Prompt Len | Output Tok | Latency (s)          | Tok/s                | Avg ITL (ms)            | P50 ITL (ms)            | P90 ITL (ms)            | Peak Mem (MB)              | Delta Mem (MB)          | MFU (%)           |
| ----------:| ----------:| --------------------:| --------------------:| -----------------------:| -----------------------:| -----------------------:| --------------------------:| -----------------------:| -----------------:|
| 32         | 32         | 0.798 [0.752, 0.824] | 40.13 [38.84, 42.55] | 24.95 [23.53, 25.76]    | 24.43 [22.72, 25.69]    | 26.90 [25.41, 28.35]    | 1405.36 [1405.36, 1405.36] | 36.35 [36.35, 36.35]    | 0.11 [0.10, 0.11] |
| 64         | 32         | 0.820 [0.813, 0.870] | 39.02 [36.78, 39.36] | 25.68 [25.42, 27.19]    | 25.52 [25.16, 27.07]    | 27.54 [27.15, 28.73]    | 1423.97 [1423.97, 1423.97] | 54.96 [54.96, 54.96]    | 0.10 [0.10, 0.10] |
| 256        | 32         | 1.102 [1.091, 1.105] | 29.03 [28.96, 29.34] | 34.60 [34.16, 34.69]    | 34.37 [33.92, 34.82]    | 36.29 [35.51, 37.36]    | 1535.63 [1535.63, 1535.63] | 166.62 [166.62, 166.62] | 0.08 [0.08, 0.08] |
| 512        | 32         | 1.607 [1.597, 1.677] | 19.91 [19.08, 20.03] | 50.48 [50.16, 52.67]    | 50.34 [50.10, 51.02]    | 51.98 [51.50, 57.60]    | 1685.33 [1685.33, 1685.33] | 316.31 [316.31, 316.31] | 0.05 [0.05, 0.05] |
| 1024       | 32         | 3.604 [3.394, 4.219] | 8.88 [7.58, 9.43]    | 113.14 [106.60, 132.58] | 107.79 [106.30, 132.78] | 127.34 [109.93, 146.25] | 1982.82 [1982.82, 1982.82] | 613.80 [613.80, 613.80] | 0.02 [0.02, 0.02] |

### Qwen3-1.7B-Base 评测结果

原始数据：[`raw/md/baseline_latency_1.7b.md`](raw/md/baseline_latency_1.7b.md) | [`raw/json/baseline_latency_1.7b.json`](raw/json/baseline_latency_1.7b.json)

| Prompt Len | Output Tok | Latency (s)          | Tok/s                | Avg ITL (ms)            | P50 ITL (ms)            | P90 ITL (ms)            | Peak Mem (MB)              | Delta Mem (MB)          | MFU (%)           |
| ----------:| ----------:| --------------------:| --------------------:| -----------------------:| -----------------------:| -----------------------:| --------------------------:| -----------------------:| -----------------:|
| 32         | 32         | 1.038 [1.016, 1.126] | 30.82 [28.43, 31.49] | 32.58 [31.84, 35.34]    | 32.47 [31.96, 34.11]    | 34.83 [34.26, 41.44]    | 3550.85 [3550.85, 3550.85] | 36.47 [36.47, 36.47]    | 0.24 [0.22, 0.24] |
| 64         | 32         | 1.025 [1.003, 1.053] | 31.22 [30.40, 31.90] | 31.99 [31.31, 32.83]    | 31.79 [30.79, 32.80]    | 33.65 [32.91, 35.00]    | 3569.52 [3569.52, 3569.52] | 55.14 [55.14, 55.14]    | 0.24 [0.23, 0.24] |
| 256        | 32         | 1.852 [1.847, 1.857] | 17.28 [17.23, 17.32] | 58.30 [58.15, 58.49]    | 58.21 [57.76, 58.46]    | 60.23 [59.31, 60.85]    | 3681.55 [3681.55, 3681.55] | 167.18 [167.18, 167.18] | 0.13 [0.13, 0.13] |
| 512        | 32         | 2.683 [2.662, 2.774] | 11.93 [11.54, 12.02] | 84.18 [83.59, 87.09]    | 83.89 [83.47, 87.16]    | 87.01 [84.74, 89.58]    | 3830.93 [3830.93, 3830.93] | 316.55 [316.55, 316.55] | 0.09 [0.09, 0.09] |
| 1024       | 32         | 5.341 [5.331, 5.387] | 5.99 [5.94, 6.00]    | 167.41 [167.09, 168.98] | 167.76 [167.11, 168.77] | 170.38 [169.92, 175.72] | 4129.69 [4129.69, 4129.69] | 615.31 [615.31, 615.31] | 0.05 [0.05, 0.05] |

### 延迟与吞吐量关键观察

1. **吞吐量断崖式下跌（无 KV Cache 的 $O(N^2)$ 计算瓶颈）：**
   - 0.6B 模型：Prompt 从 32 增加到 1024，吞吐量从 **40.13 tok/s 锐减至 8.88 tok/s**（衰减 **78%**，单 token ITL 从 24.95 ms 膨胀至 113.14 ms）。
   - 1.7B 模型：Prompt 从 32 增加到 1024，吞吐量从 **30.82 tok/s 锐减至 5.99 tok/s**（衰减 **80%**，单 token ITL 从 32.58 ms 膨胀至 167.41 ms）。
   - 核心原因：由于缺乏 KV Cache，自回归生成每吐出一个 token，都必须将完整历史序列（$L+t$）重新送入完整的 28 层 Transformer 重新计算全部 Query、Key、Value 与自注意力。
2. **激活显存增长规律与模型参数解耦：**
   - 0.6B 与 1.7B 模型的显存增量（Delta Mem）几乎完全一致：Prompt 32 为 **~36 MB**，Prompt 1024 为 **~614 MB**。
   - 验证了理论：在没有 KV Cache 缓存状态的情况下，显存增量纯粹由前向激活张量（`[1, seq_len, hidden_size]` 与注意力矩阵 `[1, num_heads, seq_len, seq_len]`）驱动。

---

## 2. Profiler 算子下潜与性能剖析

**测试目标：** 通过 PyTorch Profiler 捕获单次完整前向（Prefill 阶段）计算图中的底层算子执行时间、调用频次及张量显存分配情况，定位单请求生成的核心开销。

**测试配置：**
- Prompt 长度 = 512 tokens，单步前向推理（Prefill 阶段）。
- 开启张量形状追踪（`record_shapes=True`）与显存分配追踪（`profile_memory=True`）。
- 导出轻量级 Chrome Trace 时间线（单个文件仅约 **4.8 MB**，支持直接导入 [ui.perfetto.dev](https://ui.perfetto.dev) 进行交互式时间线与火焰图分析）。

### Profiler 算子耗时 Top 10

原始报告：[`raw/md/baseline_profiler_0.6b.md`](raw/md/baseline_profiler_0.6b.md) | [`raw/md/baseline_profiler_1.7b.md`](raw/md/baseline_profiler_1.7b.md)  
Chrome Trace：[`raw/traces/baseline_trace_0.6b_prompt512.json`](raw/traces/baseline_trace_0.6b_prompt512.json) | [`raw/traces/baseline_trace_1.7b_prompt512.json`](raw/traces/baseline_trace_1.7b_prompt512.json)

| Op 名称 | 0.6B Total Time | 0.6B Calls | 1.7B Total Time | 1.7B Calls | 1.7B 显存分配 | 算子物理含义与 28 层结构映射 |
|:---|---:|---:|---:|---:|---:|:---|
| `aten::is_nonzero` / `item` | 20.6 ms | 56 | 48.7 ms | 56 | 0 B | **异常设备同步点**（28 层 $\times$ 2 次 RoPE 边界断言中的 `.any()` 标量取值） |
| `aten::matmul` / `bmm` / `linear` | 5.3 ms | 338 | 6.0 ms | 338 | 1.97 GB | 核心矩阵乘法（28 层 $\times$ 4 组 QKV/FFN 线性投影 + 注意力矩阵计算） |
| `aten::to` / `_to_copy` | 5.8 ms | 1,015 | 6.4 ms | 1,015 | 1.72 GB | 数据类型转换与连续内存布局适配 |
| `aten::mul` | 2.4 ms | 506 | 2.7 ms | 506 | 1.48 GB | RoPE 旋转逐元素乘法、SwiGLU 门控乘法与残差缩放 |
| `aten::copy_` | 2.1 ms | 563 | 2.2 ms | 563 | 0 B | 内存连续化与张量原地拷贝 |
| `aten::add` | 0.89 ms | 225 | 0.91 ms | 225 | 197.4 MB | 残差连接累加（28 层 $\times$ 2 次 Residual Add）与频段相加 |
| `aten::clone` | 0.92 ms | 112 | 1.03 ms | 112 | 392.0 MB | 注意力头切分与 Q/K 转置缓冲拷贝（28 层 $\times$ 4 组头） |
| `aten::repeat_interleave` | 0.90 ms | 56 | 1.01 ms | 56 | 112.0 MB | GQA 机制中 Key/Value 头维度的广播复制（28 层 $\times$ 2 组 KV） |
| `aten::masked_fill` | 0.73 ms | 28 | 0.79 ms | 28 | 224.0 MB | 因果注意力掩码填充（28 层因果 Mask 下三角填充 $-\infty$） |
| `aten::pow` | 0.66 ms | 113 | 0.65 ms | 113 | 396.0 MB | RMSNorm 计算中的平方项运算（28 层 $\times$ 2 次 Normalization） |

### Profiler 关键发现与隐患挖掘

1. **重大发现：RoPE 边界防御性断言引入了隐式 Host-Device 强同步！**
   - 在单次前向中，`aten::is_nonzero` 与 `aten::item` 占据了总 CPU 调度耗时的 **54% ~ 72%**（0.6B 占 20.6 ms，1.7B 占 48.7 ms）。
   - **成因定位：** 在 `RotaryEmbedding.forward()` 中：
     ```python
     if (positions < 0).any() or (positions >= self.max_position_embeddings).any():
         raise ValueError(...)
     ```
     `(positions < 0).any()` 会在 GPU 上生成一个布尔 Tensor，而 Python 的 `if` 分支判断为了求值该布尔值，隐式触发了 GPU 同步等待（Device-to-Host 数据拷贝）。
   - **影响面：** 每个 DecoderLayer 都会调用 RoPE，单步前向内即发生 **56 次**强制设备阻塞，严重打断了 GPU Stream 的异步发射流水线。这属于未来代码优化（如将断言移至外部或改为编译期检查）的重点提升项。
2. **计算图调用次数与 28 层 Transformer 架构完美对齐：**
   - 单步采样后，算子调用次数变得极其清晰透明：
     - `aten::masked_fill` 正好调用 **28 次**（每层 1 次因果掩码）；
     - `aten::repeat_interleave` 正好调用 **56 次**（每层 2 次，分别对 Key 和 Value 做 GQA 广播扩展）；
     - `aten::linear` 正好调用 **113 次**（28 层 $\times$ 4 个投影 + 1 个 lm_head）；
     - `aten::pow` 正好调用 **113 次**（28 层 $\times$ 2 个 RMSNorm + 1 个 final norm）。
3. **轻量化的时间线工件：**
   - 单步前向 Trace 文件体积仅约 **4.8 MB**，便于在版本控制与实验归档中持久化保存，可直接导入 [ui.perfetto.dev](https://ui.perfetto.dev) 进行交互式时间线与火焰图分析。

---

## 3. Key Takeaways（核心认知与复盘）

### GPU 繁忙度 ≠ 计算效率（MFU 极低的本质）

- **现象：** 在整个生成过程中，`nvidia-smi` 观测到的 GPU 算力利用率（GPU-Util）高达 **98%~99%**，但计算得出的 MFU 仅有 **0.02% ~ 0.24%**。
- **理论推导：**
  ```text
  MFU = (2 × N_params × Throughput) / (Peak TFLOPS × 10^12) × 100%

  以 1.7B 模型在 Prompt 32 时的表现为例：
  MFU = (2 × 1.72 × 10^9 × 30.82) / (45.0 × 10^12) × 100% = 0.235% ≈ 0.24%
  ```
- **根因解析：**
  - **Memory-Bound 限制：** 自回归单请求生成在本质上是**矩阵-向量乘法（GEMV）**，算术强度（Arithmetic Intensity）只有 $\sim 1 \text{ FLOP/Byte}$。每一生成步都需要把数十亿参数从显存（HBM）搬运到 SM 缓存中，GPU 的大部分计算单元都在饥饿等待显存带宽，而非执行有效浮点运算。
  - **无 Batching：** `batch_size=1` 无法摊薄权重搬运的访存开销。

### 当前阶段（Day 9）性能瓶颈诊断

```text
[Day 9 现状]
1. 缺乏 KV Cache     ──> 每步全序列 O(N²) 重复前向 ──> 吞吐随序列长度衰减 80%
2. 缺乏 Batching      ──> 单请求访存带宽受限 (GEMV)   ──> MFU 低于 0.3%
3. 动态显存频繁申请   ──> 缺少预分配池化管理机制     ──> 频繁触发 PyTorch 显存分配器
4. 逐层布尔值同步     ──> RoPE 存在 Host 同步屏障    ──> GPU 异步发射流水线受挫
```

### 未来阶段优化基线对照表 (The Baseline to Beat)

| 核心指标                         | Qwen3-0.6B 基线          | Qwen3-1.7B 基线          | 下一阶段（KV Cache）预期目标                |
|:-------------------------------- |:------------------------ |:------------------------ |:------------------------------------------------- |
| **短序列吞吐 (Len 32)**          | 40.1 tok/s               | 30.8 tok/s               | 消除重复前向，单步 ITL 保持恒定                   |
| **长序列吞吐 (Len 1024)**        | **8.9 tok/s**            | **6.0 tok/s**            | **提升 3x ~ 5x+**，彻底消除长序列断崖衰减         |
| **单 token 解码 ITL (Len 1024)** | **113.1 ms**             | **167.4 ms**             | **压缩至 20~30 ms**，复杂度从 $O(N)$ 降至常数级别 |
| **显存增长机制**                 | 无状态激活峰值 (+614 MB) | 无状态激活峰值 (+615 MB) | 引入显式 `[batch, heads, seq, dim]` 状态缓存      |
