import pytest

from stepinfer.config.model_config import ModelConfig, load_model_config


def test_loads_default_model():
    config = load_model_config()

    assert isinstance(config, ModelConfig)
    assert config.name == "qwen3-0.6b"
    assert config.model_dir.is_dir()


@pytest.mark.parametrize(
    ("model_name", "directory_name"),
    [
        ("qwen3-0.6b", "Qwen3-0.6B-Base"),
        ("qwen3-1.7b", "Qwen3-1.7B-Base"),
    ],
)
def test_loads_explicit_model(model_name, directory_name):
    config = load_model_config(model_name)

    assert config.name == model_name
    assert config.model_dir.name == directory_name
    assert config.model_dir.is_dir()


def test_missing_model_raises_value_error():
    with pytest.raises(ValueError, match="Model not found"):
        load_model_config("missing-model")


def test_missing_config_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model config file does not exist"):
        load_model_config(config_path=tmp_path / "missing.yaml")

