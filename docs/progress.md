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
