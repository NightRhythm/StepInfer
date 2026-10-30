"""从 YAML 配置文件读取并校验模型设置。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """一个已选模型的配置。"""

    name: str
    model_dir: Path


def project_root() -> Path:
    """根据当前文件的位置返回项目根目录。"""

    # model_config.py -> config -> stepinfer -> src -> 项目根目录
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件，并进行基础结构校验。"""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Model config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("The root of the model config must be a dictionary.")

    return data


def load_model_config(
    model_name: str | None = None,
    config_path: str | Path = "configs/models.yaml",
) -> ModelConfig:
    """读取指定模型的配置。

    YAML 文件大致应具有以下结构：

        models:
          qwen3-0.6b:
            model_dir: assets/models/Qwen3-0.6B-Base
    """

    data = load_yaml(config_path)

    if "models" not in data:
        raise ValueError("Model config is missing the 'models' section.")

    models = data["models"]
    if not isinstance(models, dict):
        raise ValueError("The 'models' section must be a dictionary.")

    # 支持默认模型回退
    if model_name is None:
        if "default_model" not in data:
            raise ValueError("No 'model_name' provided and no 'default_model' specified in config.")
        model_name = data["default_model"]


    if model_name not in models:
        raise ValueError(f"Model not found in config: {model_name}")

    model_conf = models[model_name]
    if not isinstance(model_conf, dict):
        raise ValueError(f"Configuration for model '{model_name}' must be a dictionary.")

    if "model_dir" not in model_conf:
        raise ValueError(f"Configuration for model '{model_name}' is missing 'model_dir'.")

    model_dir_value = model_conf["model_dir"]
    if not isinstance(model_dir_value, (str, Path)):
        raise ValueError(f"'model_dir' for model '{model_name}' must be a path string.")

    model_dir = Path(model_dir_value)

    if not model_dir.is_absolute():
        model_dir = project_root() / model_dir

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    return ModelConfig(name=model_name, model_dir=model_dir) # type: ignore


