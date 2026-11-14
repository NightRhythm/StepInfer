import pytest

from stepinfer.config.model_config import ModelConfig, load_model_config


def test_loads_model_config():
    """验证默认配置与显式指定模型名称的配置加载。"""
    config_default = load_model_config()
    assert config_default.name == "qwen3-0.6b"
    assert config_default.model_dir.is_dir()

    config_explicit = load_model_config("qwen3-1.7b")
    assert config_explicit.name == "qwen3-1.7b"
    assert config_explicit.model_dir.name == "Qwen3-1.7B-Base"


def test_missing_model_raises_value_error():
    with pytest.raises(ValueError, match="Model not found"):
        load_model_config("missing-model")


def test_missing_config_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model config file does not exist"):
        load_model_config(config_path=tmp_path / "missing.yaml")

