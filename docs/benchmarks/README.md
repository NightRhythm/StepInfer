# StepInfer 基准测试规范与组织架构

本项目遵循单一职责与数据/报告解耦原则，将性能测量脚本与原始工件清晰组织。

---

## 1. 目录结构与工件归档

```text
docs/benchmarks/
├── baseline_benchmark.md        # 人工综合评审的主线基线报告（含深度复盘与分析）
├── README.md                    # 本测试说明指南
└── raw/                         # 自动化测试导出的原始数据与工件
    ├── md/                      # 纯净的 Markdown 格式测量数据表（无冗余声明）
    │   ├── baseline_latency_0.6b.md
    │   ├── baseline_latency_1.7b.md
    │   ├── baseline_profiler_0.6b.md
    │   └── baseline_profiler_1.7b.md
    ├── json/                    # 完整可机器解析的原始统计与逐轮数据
    │   ├── baseline_latency_0.6b.json
    │   └── baseline_latency_1.7b.json
    └── traces/                  # Chrome Trace 时间线（支持 Perfetto / Chrome 交互剖析）
        ├── baseline_trace_0.6b_prompt512.json
        └── baseline_trace_1.7b_prompt512.json
```

---

## 2. 测量脚本与运行方式

### 2.1 延迟与吞吐量基准 (`benchmarks/latency_benchmark.py`)

专用于高精度的端到端生成延迟、吞吐率、显存分配及 MFU 测量，**完全隔离 Profiler 开销**。

```bash
# 评估 Qwen3-0.6B-Base
uv run python benchmarks/latency_benchmark.py --model-path assets/models/Qwen3-0.6B-Base --tag 0.6b

# 评估 Qwen3-1.7B-Base
uv run python benchmarks/latency_benchmark.py --model-path assets/models/Qwen3-1.7B-Base --tag 1.7b
```

主要参数：
- `--prompt-lengths`: 输入序列长度列表，默认为 `32 64 256 512 1024`。
- `--max-new-tokens`: 固定生成的 token 数，默认为 `32`。
- `--warmup-runs`: 预热轮数，默认为 `3`。
- `--repeats`: 正式测量轮数，默认为 `10`。
- `--tag`: 产物文件标签（如 `0.6b`、`1.7b`），输出至 `raw/md/latency_<tag>.md` 和 `raw/json/latency_<tag>.json`。

### 2.2 算子剖析与时间线分析 (`benchmarks/profiler_benchmark.py`)

专用于单次典型序列推理的算子调用分解与计算图追踪，捕获算子耗时与显存事件，导出 Chrome Trace 时间线。

```bash
# 剖析 Qwen3-0.6B-Base (Prompt 512, 单步前向)
uv run python benchmarks/profiler_benchmark.py --model-path assets/models/Qwen3-0.6B-Base --tag 0.6b --prompt-length 512

# 剖析 Qwen3-1.7B-Base (Prompt 512, 单步前向)
uv run python benchmarks/profiler_benchmark.py --model-path assets/models/Qwen3-1.7B-Base --tag 1.7b --prompt-length 512
```

产物说明：
- Trace 文件：输出至 `raw/traces/trace_<tag>_prompt<len>.json`，可直接拖入 [ui.perfetto.dev](https://ui.perfetto.dev) 进行交互式时间线定位与火焰图分析。
- 剖析表格：输出至 `raw/md/profiler_<tag>.md`，包含 Top 15 算子耗时与显存开销排序。

> **归档规范**：测试脚本默认生成标准的通用基础工件（`latency_*`、`profiler_*`、`trace_*`）。在阶段评审与报告归档时，人工重命名加上对应阶段前缀（如 `baseline_`、后续阶段的 `kv_cache_` 等），保持代码生成与归档记录的彻底解耦。

---

## 3. 指标口径与测量原则

- **测量范围：** 仅包含“设备上常驻输入 token ID → 固定步数自回归生成循环”，排除了分词（Tokenize）、主机到设备拷贝（Host-to-Device Copy）和文本解码（Decode）的干扰。
- **时间度量：** CUDA 模式下使用 `torch.cuda.Event` 精确度量首 token 间隔与逐步 ITL；吞吐量基于包含流同步的端到端生成墙钟耗时计算。
- **MFU 计算口径：**
  ```text
  MFU = (2 × num_params × throughput) / (peak_tflops × 10^12) × 100%
  ```
- **显存度量：** 记录 PyTorch 显存分配器的已分配峰值（Peak Allocated）与相对于起点的显存增量（Peak Delta）。
