"""通用权重加载器单元测试，涵盖 Safetensors 读取、Merged GEMM 拼接映射、权重对齐审计与设备迁移。"""

from pathlib import Path
import pytest
import torch

from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.models import Qwen3ForCausalLM, load_model_weights


@pytest.fixture(scope="module")
def real_model_path() -> Path:
    """返回本地真实的 Qwen3-0.6B-Base 权重路径。"""
    path = Path("assets/models/Qwen3-0.6B-Base").resolve()
    if not (path / "model.safetensors").exists():
        pytest.skip("Local model.safetensors not found, skipping real weights test.")
    return path


def test_load_model_weights_full_qwen3_success(real_model_path):
    """验证从真实 safetensors 加载 Qwen3-0.6B 权重，310 个张量 100% 对齐无遗漏。"""
    config = Qwen3Config.from_json_file(real_model_path / "config.json")
    model = Qwen3ForCausalLM(config)

    result = load_model_weights(
        model=model,
        model_path=real_model_path,
        device="cpu",
        dtype=torch.float32,
        strict=True,
    )

    # 1. 核心体检审计断言
    assert result.is_success
    assert result.total_checkpoint_tensors == 310
    assert result.loaded_checkpoint_tensors == 310
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0
    assert len(result.shape_mismatches) == 0

    # 2. 权重绑定 (Weight Tying) 指针共享断言
    assert model.lm_head.weight is model.model.embed_tokens.weight

    # 3. 参数赋值非零检查 (确认真实权重已覆盖初始值)
    assert model.model.embed_tokens.weight.abs().sum() > 0
    assert model.model.layers[0].self_attn.qkv_proj.weight.abs().sum() > 0
    assert model.model.layers[0].mlp.gate_up_proj.weight.abs().sum() > 0

    # 4. 报告字符串输出检查
    summary = result.format_summary()
    assert "所有参数严丝合缝对齐就绪" in summary


def test_load_model_weights_path_validation(tmp_path: Path):
    """验证路径不存在、格式非法时的异常防御。"""
    config = Qwen3Config(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(config)

    # 不存在的路径
    with pytest.raises(FileNotFoundError, match="Weight path does not exist"):
        load_model_weights(model, tmp_path / "non_existent_dir")

    # 非 .safetensors 文件
    fake_file = tmp_path / "test.bin"
    fake_file.write_text("dummy")
    with pytest.raises(ValueError, match="not a .safetensors file"):
        load_model_weights(model, fake_file)

    # 空目录 (无 .safetensors)
    with pytest.raises(FileNotFoundError, match="No .safetensors files found"):
        load_model_weights(model, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_load_model_weights_gpu_migration(real_model_path):
    """验证将真实模型权重直接加载并迁移至宿主机 GPU (RTX 5060 Ti)。"""
    config = Qwen3Config.from_json_file(real_model_path / "config.json")
    model = Qwen3ForCausalLM(config)

    result = load_model_weights(
        model=model,
        model_path=real_model_path,
        device="cuda",
        dtype=torch.bfloat16,
        strict=True,
    )

    assert result.is_success
    # 确认参数已放置在 GPU 设备上
    assert next(model.parameters()).device.type == "cuda"
    assert next(model.parameters()).dtype == torch.bfloat16
