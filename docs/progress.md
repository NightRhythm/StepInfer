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