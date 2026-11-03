"""Week 1 / Day 3 冒烟测试: Qwen3 自定义 Transformer 模型接入 InferenceEngine 端到端前向推理验证。"""

import pytest
import torch

from stepinfer.config.model_config import load_model_config
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.models.qwen3 import Qwen3ForCausalLM
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    """加载本地真实的 Qwen3 Tokenizer。"""
    return HFTokenizer(load_model_config())


@pytest.fixture
def tiny_qwen3_model(qwen3_tokenizer):
    """构造轻量 2 层 Qwen3 模型用于秒级快速冒烟测试。"""
    config = Qwen3Config.from_json_file("assets/models/Qwen3-0.6B-Base/config.json")
    config.num_hidden_layers = 2
    return Qwen3ForCausalLM(config)


def test_qwen3_inference_engine_smoke_generation(qwen3_tokenizer, tiny_qwen3_model):
    """验证自定义 Qwen3 模型成功接入 InferenceEngine，端到端完成 Token 自回归生成。"""
    engine = InferenceEngine(
        model=tiny_qwen3_model,
        tokenizer=qwen3_tokenizer,
        device="cpu",
    )

    prompt = "Hello"
    max_new_tokens = 3
    result = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)

    assert result["prompt"] == prompt
    assert isinstance(result["generated_text"], str)
    assert len(result["generated_text"]) > len(prompt)
    assert result["token_count"] == max_new_tokens
    assert result["stop_reason"] == f"Reached max_new_tokens limit of {max_new_tokens}."
    assert result["sampling_method"] == "greedy"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_qwen3_inference_engine_gpu_execution(qwen3_tokenizer, tiny_qwen3_model):
    """验证宿主机真实 GPU (NVIDIA RTX 5060 Ti) 上的端到端前向推理与生成。"""
    engine = InferenceEngine(
        model=tiny_qwen3_model,
        tokenizer=qwen3_tokenizer,
        device="cuda",
    )

    prompt = "StepInfer"
    result = engine.generate(prompt=prompt, max_new_tokens=2)

    assert result["prompt"] == prompt
    assert isinstance(result["generated_text"], str)
    assert result["token_count"] == 2
