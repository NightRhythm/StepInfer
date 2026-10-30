import pytest
import torch

from stepinfer.config.model_config import load_model_config
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return HFTokenizer(load_model_config())


def test_encode_returns_list_for_default_python_output(tokenizer):
    token_ids = tokenizer.encode("你好，stepinfer", return_tensors=None)

    assert isinstance(token_ids, list)
    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_encode_returns_tensor_for_pt_output(tokenizer):
    token_ids = tokenizer.encode("你好，stepinfer")

    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.ndim == 2
    assert token_ids.shape[0] == 1
    assert token_ids.shape[1] > 0


def test_encode_decode_round_trip_with_list(tokenizer):
    text = "你好，stepinfer"
    token_ids = tokenizer.encode(text, return_tensors=None)

    assert tokenizer.decode(token_ids) == text


def test_encode_decode_round_trip_with_tensor(tokenizer):
    text = "你好，stepinfer"
    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text


def test_special_token_ids_are_exposed(tokenizer):
    assert isinstance(tokenizer.eos_token_id, int)
    assert tokenizer.bos_token_id is None or isinstance(tokenizer.bos_token_id, int)
    assert tokenizer.pad_token_id is None or isinstance(tokenizer.pad_token_id, int)


def test_batch_text_is_not_supported_yet(tokenizer):
    with pytest.raises(TypeError, match="single string only"):
        tokenizer.encode(["first", "second"])

