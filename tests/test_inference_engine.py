import pytest
import torch
from torch import nn

from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.tokenizer.base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    @property
    def eos_token_id(self):
        return 2

    def encode(self, text, **kwargs):
        assert kwargs.get("return_tensors") == "pt"
        return torch.tensor([[ord(char) for char in text]], dtype=torch.long)

    def decode(self, token_ids, **kwargs):
        if token_ids.ndim == 2:
            token_ids = token_ids[0]
        return "".join(chr(token_id) for token_id in token_ids.tolist())


class ConstantLogitsModel(nn.Module):
    def __init__(self, next_token_id: int):
        super().__init__()
        self.next_token_id = next_token_id

    def forward(self, input_ids):
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], 128, device=input_ids.device
        )
        logits[..., self.next_token_id] = 1.0
        return logits


def test_engine_generates_text_and_uses_tokenizer_eos_id():
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
    )

    result = engine.generate("hi", max_new_tokens=2)

    assert engine.eos_token_id == 2
    assert result == {
        "prompt": "hi",
        "generated_text": "hi!!",
        "sampling_method": "greedy",
        "token_count": 2,
        "stop_reason": "Reached max_new_tokens limit of 2.",
    }


def test_engine_uses_explicit_eos_id_in_preference_to_tokenizer():
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
        eos_token_id=99,
    )

    assert engine.eos_token_id == 99


def test_engine_sets_model_to_evaluation_mode():
    model = ConstantLogitsModel(next_token_id=33)
    model.train()

    engine = InferenceEngine(model=model, tokenizer=CharacterTokenizer())

    assert engine.model.training is False


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_engine_rejects_empty_or_whitespace_prompt(prompt):
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
    )

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        engine.generate(prompt)


def test_engine_rejects_non_string_prompt():
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
    )

    with pytest.raises(TypeError, match="prompt must be a string"):
        engine.generate(123)


def test_engine_rejects_non_positive_generation_limit():
    engine = InferenceEngine(
        model=ConstantLogitsModel(next_token_id=33),
        tokenizer=CharacterTokenizer(),
    )

    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        engine.generate("hi", max_new_tokens=0)


def test_engine_rejects_non_module_model():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        InferenceEngine(model=object(), tokenizer=CharacterTokenizer())

