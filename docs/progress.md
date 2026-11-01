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
- 🔄 推进中：待实现 RoPE、SwiGLU 与 Attention 核心模块