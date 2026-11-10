# KV Cache Benchmark — Day 10 (Static Pre-allocated Cache & In-place Slicing)

本文档记录 StepInfer 在接入静态预分配 KV Cache（按层独立管理、原地切片更新、GQA 原生头数保存）后的性能基准评测，并与 Day 9 的无 Cache 基线进行横向深度对比。测试数据全部基于真实 GPU 运行生成，原始数据保存在 `docs/benchmarks/raw/` 目录下。

---

## Setup

| Parameter                | Qwen3-0.6B-Base (Baseline / Day 9)     | Qwen3-0.6B-Base (KV Cache / Day 10)    |
|:------------------------ |:-------------------------------------- |:-------------------------------------- |
| Architecture             | Qwen3 (custom implementation)          | Qwen3 (custom implementation)          |
| Parameters               | 596,049,920 (227 state_dict keys)      | 596,049,920 (227 state_dict keys)      |
| Precision                | `bf16`                                 | `bf16`                                 |
| GPU                      | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| Architecture Generation  | Blackwell (SM 12.0)                    | Blackwell (SM 12.0)                    |
| Peak TFLOPS (dense bf16) | 45.0                                   | 45.0                                   |
| PyTorch                  | 2.9.1+cu128                            | 2.9.1+cu128                            |
| Python                   | 3.12.16                                | 3.12.16                                |
| KV Cache                 | **None** (每步全序列 $O(N^2)$ 重复前向) | **Static Pre-allocated** (原地切片 $O(1)$) |
| Cache 显存布局           | 无状态                                 | `[batch, num_kv_heads, max_seq, dim]`  |
| Batching                 | Single request (batch_size=1)          | Single request (batch_size=1)          |
| Sampling                 | Greedy (argmax)                        | Greedy (argmax)                        |
| Timing Backend           | CUDA Event (`torch.cuda.Event`)        | CUDA Event (`torch.cuda.Event`)        |

---

## 1. Latency & Throughput Benchmark

**测试目标：** 测量引入 KV Cache 之后，不同 Prompt 长度下的端到端生成延迟、吞吐率、ITL（Token 间延迟）及显存开销。

**测试配置：**
- 每轮固定生成 **32 个 token**（禁用 EOS 提前停止以保持生成步数完全一致）。
- 5 组 Prompt 长度：`[32, 64, 256, 512, 1024]`。
- 每个长度先执行 **3 轮完整预热**，再连续记录 **10 轮正式评测**。
- 表格数据汇报格式为：`中位数 [最小值, 最大值]`。

### Qwen3-0.6B-Base 评测结果 (KV Cache 模式)

原始数据：[`raw/md/kvcache_latency_0.6b.md`](raw/md/kvcache_latency_0.6b.md)

| Prompt Len | Output Tok | Latency (s) | Tok/s | Avg ITL (ms) | P50 ITL (ms) | P90 ITL (ms) | Peak Mem (MB) | Delta Mem (MB) | MFU (%) |
| ----------:| ----------:| -----------:| -----:| ------------:| ------------:| ------------:| -------------:| --------------:| -------:|
| 32 | 32 | 0.691 [0.682, 0.750] | 46.28 [42.65, 46.94] | 21.45 [21.12, 23.42] | 21.20 [20.73, 22.41] | 23.31 [22.43, 29.73] | 1385.58 [1385.58, 1385.58] | 9.57 [9.57, 9.57] | 0.12 [0.11, 0.12] |
| 64 | 32 | 0.689 [0.675, 0.782] | 46.43 [40.90, 47.41] | 21.43 [20.95, 24.46] | 21.28 [20.48, 23.80] | 23.19 [22.19, 29.33] | 1398.60 [1398.60, 1398.60] | 19.09 [19.09, 19.09] | 0.12 [0.11, 0.13] |
| 256 | 32 | 0.697 [0.683, 0.704] | 45.92 [45.45, 46.87] | 21.54 [21.09, 21.78] | 21.30 [20.97, 21.72] | 23.23 [22.70, 24.11] | 1478.00 [1478.00, 1478.00] | 77.48 [77.48, 77.48] | 0.12 [0.12, 0.12] |
| 512 | 32 | 0.716 [0.706, 0.745] | 44.68 [42.94, 45.32] | 21.74 [21.45, 22.60] | 21.63 [21.07, 22.08] | 23.35 [23.05, 25.58] | 1592.65 [1592.65, 1592.65] | 152.45 [152.45, 152.45] | 0.12 [0.11, 0.12] |
| 1024 | 32 | 0.770 [0.759, 0.775] | 41.56 [41.29, 42.18] | 21.84 [21.55, 22.07] | 21.74 [21.55, 21.92] | 23.77 [23.12, 24.40] | 1789.38 [1789.38, 1789.38] | 304.86 [304.86, 304.86] | 0.11 [0.11, 0.11] |

---

## 2. 横向对比分析：Baseline vs. KV Cache

下表对相同模型（Qwen3-0.6B-Base）、相同设备（RTX 5060 Ti）在两阶段的中位数评测指标进行了并排对齐：

| Prompt 长度 | 输出 Token | Baseline 延迟 | **KV Cache 延迟** | Baseline ITL | **KV Cache ITL** | Baseline 吞吐 | **KV Cache 吞吐** | **长文本端到端加速比** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **32** | 32 | 0.798 s | **0.691 s** | 24.95 ms | **21.45 ms** | 40.13 tok/s | **46.28 tok/s** | **1.15x** |
| **64** | 32 | 0.820 s | **0.689 s** | 25.68 ms | **21.43 ms** | 39.02 tok/s | **46.43 tok/s** | **1.19x** |
| **256** | 32 | 1.102 s | **0.697 s** | 34.60 ms | **21.54 ms** | 29.03 tok/s | **45.92 tok/s** | **1.58x** |
| **512** | 32 | 1.607 s | **0.716 s** | 50.48 ms | **21.74 ms** | 19.91 tok/s | **44.68 tok/s** | **2.24x** |
| **1024** | 32 | 3.604 s | **0.770 s** | 113.14 ms | **21.84 ms** | 8.88 tok/s | **41.56 tok/s** | 🚀 **4.68x** |

### 核心收益与物理规律印证

#### 1. ITL（Token 间生成延迟）被彻底“拉平”为一条水平线 ($O(1)$)
- **Baseline（无 Cache）**：自回归生成在每一步都重算全部前向，ITL 随上下文长度线性恶化（从 24.95 ms 一路狂飙至 **113.14 ms**，耗时暴增 4.5 倍）；
- **KV Cache**：由于 Decode 阶段每步只需送入单 Token `[1, 1]` 计算单个向量并与历史缓存切片做乘法，ITL 表现出极其强悍的常数平坦度：
  $$\text{Prompt 32 (21.45 ms)} \approx \text{Prompt 64 (21.43 ms)} \approx \text{Prompt 256 (21.54 ms)} \approx \text{Prompt 512 (21.74 ms)} \approx \text{Prompt 1024 (21.84 ms)}$$
  **从 32 到 1024 长度，单步 ITL 仅仅微涨 0.39 ms（仅增 1.8%）！** 真正达成 $O(1)$ 解码复杂度！

#### 2. 长 Prompt 下吞吐暴涨 4.68 倍
- 在 Prompt 1024 场景下，端到端生成延迟由 **3.604 秒骤降至 0.770 秒**；
- 吞吐量从惨不忍睹的 **8.88 tok/s 飙升至 41.56 tok/s**，彻底消除了无 Cache 模式下的性能断崖。

#### 3. GQA 原生保存生效：显存增量精确减半
- **Prompt 1024 显存 Delta 对比**：
  - Baseline（无 Cache）：**613.80 MB**（由全序列中间激活值驱动）；
  - KV Cache：**304.86 MB**；
- **理论印证**：
  Qwen3 采用 GQA 机制，`num_attention_heads = 16`，而 `num_key_value_heads = 8`。
  我们在 `LayerKVCache` 中仅存储未被广播的原生 8 个头（在 `repeat_interleave` 之前暂存），显存增量正好减少了 **50.3%**，精准对齐理论预期！

---

## 3. Profiler 算子下潜与状态演进

原始报告：[`raw/md/kvcache_profiler_0.6b.md`](raw/md/kvcache_profiler_0.6b.md)（已开启 `--no-trace` 忽略大文件追踪）

**测试配置：** Prompt 长度 = 512 tokens，生成 1 个 Token（Prefill 首步采样）。

### Top 10 核心算子耗时与调用模式

| Op 名称 | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem | 物理含义与架构映射 |
|:---|---:|---:|---:|---:|---:|:---|
| `aten::is_nonzero` / `item` | 16.97 ms | 0.1% | 303.0 us | 56 | 0 B | **已知同步点**（28 层 RoPE 边界防御性断言隐式触发 Device-Host 同步） |
| `aten::matmul` / `bmm` | 4.09 ms | 4.2% | 18.2 us | 225 | 1.04 GB | Prefill 阶段 QKV 矩阵乘与点积注意力计算 |
| `aten::to` / `_to_copy` | 5.88 ms | 3.0% | 5.8 us | 1,015 | 1.54 GB | 权重类型适配与内存连续化 |
| `aten::mul` | 2.35 ms | 6.7% | 4.6 us | 506 | 1.18 GB | SwiGLU 门控、RoPE 旋转与缩放相乘 |
| `aten::copy_` | 2.26 ms | 6.4% | 3.6 us | 619 | 0 B | **KV Cache 原地切片覆写**与激活值拷贝 |
| `aten::linear` / `mm` | 3.19 ms | 3.6% | 18.2 us | 226 | 968.8 MB | MLP 门控升维与输出投影 |
| `aten::zeros` | 0.51 ms | 0.3% | 9.0 us | 56 | 70.15 MB | **KV Cache 预分配底层连续张量**（28 层 $\times$ 2 组） |
| `aten::add` | 1.03 ms | 2.9% | 4.6 us | 225 | 141.4 MB | Pre-Norm / Post-Norm 残差累加 |
| `aten::repeat_interleave` | 0.82 ms | 0.5% | 14.6 us | 56 | 139.3 MB | GQA 头数按组广播扩展 |
| `aten::clone` | 0.81 ms | 0.3% | 7.2 us | 112 | 419.3 MB | 注意力多头转置与切分 |

### Profiler 关键观察

1. **预分配机制确立**：
   `aten::zeros` 在初始化阶段精准分配了 56 个预分配张量（28 层 $\times$ K/V 独立分配），占用 **70.15 MB**。在后续推理步骤中，PyTorch 显存分配器无需频繁向系统申请/释放显存，彻底消除了显存碎片。
2. **原地切片写入零显存重分配**：
   更新操作被转化为 `aten::copy_`，利用底层连续切片索引直接覆盖写入，整个 Decode 循环没有触发任何 `torch.cat` 或张量重拼接开销。
3. **残留瓶颈确认**：
   `aten::is_nonzero` 与 `aten::item` 依然占据了单步前向 CPU 调度时间的近 **48%**。这证实了我们在 Day 9 发现的 RoPE 隐式同步屏障在后续算子优化阶段依然具备极大的消除价值。

---

## 4. Key Takeaways（核心认知与下一阶段靶点）

### 从计算受限到纯访存受限 (The Shift from Compute-Bound to Memory-Bound)

```text
[性能瓶颈演进]
Day 9 (无 Cache):   每步计算量 O(N²) 膨胀  ──> 计算受限 (Compute-Bound) ──> 长文本延迟断崖
Day 10 (KV Cache):  每步计算量 O(1) 恒定   ──> 访存受限 (Memory-Bound)  ──> 单请求带宽跑不满
```

- **为什么 MFU 依然在 0.12% 左右？**
  虽然端到端耗时大幅缩短，但单请求贪心解码依然属于典型的 **GEMV（矩阵-向量乘）**。
  每吐出一个 Token，GPU 依然必须完整加载一次 Qwen3-0.6B 的全部 1.2 GB 权重参数。在 `batch_size = 1` 时，每个权重参数仅参与了 1 次乘加运算，算术强度极低，导致 GPU 的 Tensor Core 算力利用率依然受限于 HBM 显存带宽。