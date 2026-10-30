# Models

模型文件放在这个目录下。

当前项目有以下模型：

- `Qwen3-1.7B-Base`
- `Qwen3-0.6B-Base`

模型权重不会提交到 Git。可在项目根目录执行：

```bash
python scripts/download_model.py Qwen/Qwen3-1.7B-Base
python scripts/download_model.py Qwen/Qwen3-0.6B-Base
```