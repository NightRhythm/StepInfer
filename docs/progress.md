# develop progress

## Step 1

### day 1

- ✅ 设计项目架构
- ✅ 配置 Python 环境和项目依赖
- ✅ 初始化项目 `src` 目录结构

### day 2

- ✅ 编写模型下载脚本 `scripts/download_model.py`
- ✅ 下载 Qwen3-0.6B-Base 和 Qwen3-1.7B-Base 模型
- ✅ 创建模型说明文件 `assets/models/README.md`
- ✅ 编写模型配置文件 `configs/models.yaml`
- ✅ 支持多个模型和默认模型

### day 3

- ✅ 实现模型配置加载 `src/stepinfer/config/model_config.py`
- ✅ 实现 tokenizer 基础接口和 Hugging Face 适配器
- ✅ 支持单条文本编码与解码
- ✅ 编写配置和 tokenizer 测试
- ⬜ 暂不支持 batch 文本处理

### day 4

- ✅ 实现 `src/stepinfer/inference/sampler.py`（基于 argmax 的贪心采样）
- ✅ 实现 `src/stepinfer/inference/generator.py`（自回归生成：前向计算 → 采样 → EOS/最大长度停止）
- ✅ 实现 `src/stepinfer/inference/inference_engine.py`（推理协调：编码 → 生成 → 解码）
- ✅ 编写相关测试（采样、生成循环和推理协调器）
- ⬜ 暂不支持 batch、KV Cache 和其他采样策略

### day 5

- ✅ 搭建 Qwen3 现代模型架构与算子层骨架（`layers/`、`models/`）
- ✅ 实现 `src/stepinfer/config/qwen3_config.py`（超参映射、GQA 整除约束、head_dim 回退与正数防御校验）
- ✅ 实现 `src/stepinfer/layers/norm.py`（基础 RMSNorm 及残差融合 Fused Add + RMSNorm 算子）
- ✅ 编写配置与 RMSNorm 单元测试，全量通过

### day 6

- ✅ 实现 `src/stepinfer/layers/rotary_embedding.py`（RoPE 旋转位置编码，支持 Half-Split 与 GQA 广播）
- ✅ 实现 `src/stepinfer/layers/activation.py`（SwiGLU 门控前馈网络所需的 `SiluAndMul` 激活算子）
- ✅ 实现 `src/stepinfer/layers/attention.py`（带 GQA 头扩展与因果掩码的 `CausalAttention` 点积注意力算子）
- ✅ 实现 `src/stepinfer/models/qwen3.py`（Qwen3MLP、Qwen3Attention、Qwen3DecoderLayer、Qwen3Model 与 Qwen3ForCausalLM 顶层总装）
- ✅ 支持 DecoderLayer 间残差流水线融合传递与 LM Head 权重共享（Weight Tying）
- ✅ 编写算子层与模型层全量单元测试，全部通过
- ✅ 编写端到端冒烟测试 `tests/test_smoke_forward.py`，接入 `InferenceEngine` 在 CPU 与 GPU (RTX 5060 Ti) 验证自回归生成全链路通畅

### day 7

- ✅ 实现 `src/stepinfer/models/loader.py`（Safetensors 流式加载、Merged GEMM 拼接映射与对齐审计报告）
- ✅ 编写 `tests/test_weight_loader.py`（验证官方 310 个权重张量 100% 对齐就绪与 GPU 迁移）
- ✅ 编写 `tests/test_inference.py`（涵盖中英文事实常识问答、贪心解码确定性与前缀连贯性）
- ✅ 验证两类停止条件拦截（达到 `max_new_tokens` 截断与命中 EOS 提前退出）
- ⬜ 暂不支持 KV Cache（全量计算模式）与 Batch 推理

### day 8

- ✅ 实现 `src/stepinfer/utils/profiler.py`（封装 `torch.profiler`，支持 CPU/CUDA 算子剖析、张量形状与内存追踪，支持导出 Chrome Trace）
- ✅ 实现 `src/stepinfer/utils/gpu_monitor.py`（记录 PyTorch allocated/reserved 当前值与峰值，并通过 CUDA 查询设备级 free/total 显存）
- ✅ 实现 `benchmarks/metrics.py`（区分生成循环墙钟耗时与逐 token 计时区间，记录重复测量原始值及汇总统计；不再提供口径不足的 MFU 估算）
- ✅ 实现 `benchmarks/latency_benchmark.py`（固定 token-ID 输入和输出长度，支持逐 case 预热、重复测量、独立 profiler 模式与 JSON/Markdown 结果）
- ✅ 将基准运行说明和报告默认输出目录设为 `docs/benchmarks/`
- ⬜ 待在目标开发服务器上验证最新实现、确认兼容性并记录真实基线数据

### day 9

- ✅ 拆分独立的性能剖析脚本 `benchmarks/profiler_benchmark.py`（解耦算子剖析与延迟测量，优化采样策略并支持导出轻量 Chrome Trace）
- ✅ 整理延迟测试脚本 `benchmarks/latency_benchmark.py`（剥离 profiler 采样逻辑，聚焦纯净的端到端延迟、吞吐与显存测量）
- ✅ 重构基准测试指标 `benchmarks/metrics.py`（引入常见 GPU 理论峰值算力表，支持科学估算 MFU 并补充单元测试）
- ✅ 完成真实环境基线性能实测（在 RTX 5060 Ti 上完成 Qwen3-0.6B 与 1.7B 多序列长度评测并归档测试工件）
- ✅ 编写基线评测分析报告 `docs/benchmarks/baseline_benchmark.md`（汇编实测数据，深入剖析无 KV Cache 架构下的二次方计算膨胀、访存瓶颈与算子同步开销）
