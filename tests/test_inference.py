"""核心全场景验证测试 (Validation Day)。

全面验证真实预训练权重下:
1. 真实权重端到端自回归生成与事实性常识对齐;
2. 贪心解码确定性 (相同输入输出严格一致);
3. 达到 max_new_tokens 上限停止条件;
4. 命中 EOS Token 提前截断停止条件。
"""

from pathlib import Path
import pytest
import torch

from stepinfer.config.model_config import load_model_config
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def real_model_path() -> Path:
    """返回本地真实的 Qwen3-0.6B-Base 权重路径。"""
    path = Path("assets/models/Qwen3-0.6B-Base").resolve()
    if not (path / "model.safetensors").exists():
        pytest.skip("Local model.safetensors not found, skipping real model inference validation.")
    return path


@pytest.fixture(scope="module")
def shared_engine(real_model_path):
    """全局共享的真实模型推理引擎实例，优先调度 CUDA GPU 加速测试。"""
    config = Qwen3Config.from_json_file(real_model_path / "config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen3ForCausalLM(config=config)
    load_model_weights(model=model, model_path=real_model_path, device=device, dtype=dtype)
    tokenizer = HFTokenizer(load_model_config())

    return InferenceEngine(model=model, tokenizer=tokenizer, device=device)


def test_real_model_end_to_end_generation_and_factual_correctness(shared_engine):
    """验证真实预训练权重端到端推理：输出结构契约、Prompt前缀连贯性及常识生成正确性。"""
    prompt = "The capital of France is Paris, and the capital of Germany is"
    max_tokens = 6
    result = shared_engine.generate(prompt=prompt, max_new_tokens=max_tokens)

    # 1. 结构与生命周期契约
    assert result["prompt"] == prompt
    assert result["sampling_method"] == "greedy"
    assert 0 < result["token_count"] <= max_tokens
    assert result["generated_text"].startswith(prompt)

    # 2. 事实性权重正确性 (Berlin)
    assert "Berlin" in result["generated_text"]


def test_greedy_decoding_determinism(shared_engine):
    """验证贪心采样的严格确定性: 相同 Prompt 连续生成两次，输出必须逐字完全一致。"""
    prompt = "Once upon a time,"
    max_tokens = 6

    out1 = shared_engine.generate(prompt=prompt, max_new_tokens=max_tokens)
    out2 = shared_engine.generate(prompt=prompt, max_new_tokens=max_tokens)

    assert out1["generated_text"] == out2["generated_text"]
    assert out1["token_count"] == out2["token_count"]
    assert out1["stop_reason"] == out2["stop_reason"]


def test_generation_stops_at_max_tokens_limit(shared_engine):
    """验证当未产生 EOS 时，生成循环必须在达到 max_new_tokens 上限时准时截断停止。"""
    prompt = "Tell me a very long story about a magical forest"
    max_tokens = 5
    result = shared_engine.generate(prompt=prompt, max_new_tokens=max_tokens)

    assert result["token_count"] == max_tokens
    assert result["stop_reason"] == f"Reached max_new_tokens limit of {max_tokens}."


def test_generation_stops_at_eos_token(shared_engine):
    """验证自回归生成循环在遇到指定的 EOS Token ID 时能够立即提前终止。"""
    berlin_token_id = shared_engine.tokenizer.encode(" Berlin")[0, 0].item()

    eos_engine = InferenceEngine(
        model=shared_engine.model,
        tokenizer=shared_engine.tokenizer,
        device=shared_engine.device,
        eos_token_id=berlin_token_id,
    )

    prompt = "The capital of Germany is"
    result = eos_engine.generate(prompt=prompt, max_new_tokens=20)

    assert result["stop_reason"] == "Generated EOS token."
    assert result["token_count"] == 1
    assert result["generated_text"] == "The capital of Germany is Berlin"
