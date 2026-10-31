import pytest
import torch
from torch import nn

from stepinfer.inference.generator import generate_tokens


class ConstantLogitsModel(nn.Module):
    def __init__(self, next_token_id: int, vocab_size: int = 128):
        super().__init__()
        self.next_token_id = next_token_id
        self.vocab_size = vocab_size
        self.input_shapes = []
        self.grad_enabled_during_forward = []

    def forward(self, input_ids):
        self.input_shapes.append(tuple(input_ids.shape))
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[..., self.next_token_id] = 1.0
        return logits


class InvalidRankModel(nn.Module):
    def forward(self, input_ids):
        return torch.zeros(input_ids.shape[0], 10, device=input_ids.device)


class NonTensorOutputModel(nn.Module):
    def forward(self, input_ids):
        return "not logits"


def test_generator_stops_when_eos_is_generated():
    model = ConstantLogitsModel(next_token_id=2)
    input_ids = torch.tensor([[10, 11]])

    generated_ids, token_count, stop_reason = generate_tokens(
        model,
        input_ids,
        device="cpu",
        max_new_tokens=5,
        eos_token_id=2,
    )

    assert torch.equal(generated_ids, torch.tensor([[10, 11, 2]]))
    assert token_count == 1
    assert stop_reason == "Generated EOS token."


def test_generator_stops_at_max_new_tokens_and_reprocesses_full_sequence():
    model = ConstantLogitsModel(next_token_id=33)
    input_ids = torch.tensor([[10, 11]])

    generated_ids, token_count, stop_reason = generate_tokens(
        model,
        input_ids,
        device="cpu",
        max_new_tokens=3,
    )

    assert torch.equal(generated_ids, torch.tensor([[10, 11, 33, 33, 33]]))
    assert token_count == 3
    assert stop_reason == "Reached max_new_tokens limit of 3."
    assert model.input_shapes == [(1, 2), (1, 3), (1, 4)]
    assert model.grad_enabled_during_forward == [False, False, False]


@pytest.mark.parametrize(
    ("input_ids", "max_new_tokens", "error_type"),
    [
        ("not a tensor", 2, TypeError),
        (torch.empty((1, 0), dtype=torch.long), 2, ValueError),
        (torch.empty((2, 3), dtype=torch.long), 2, ValueError),
        (torch.tensor([[1, 2]]), 0, ValueError),
    ],
)
def test_generator_validates_inputs(input_ids, max_new_tokens, error_type):
    with pytest.raises(error_type):
        generate_tokens(
            ConstantLogitsModel(next_token_id=33),
            input_ids,
            device="cpu",
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize(
    ("model", "error_type"),
    [
        (InvalidRankModel(), ValueError),
        (NonTensorOutputModel(), TypeError),
    ],
)
def test_generator_validates_model_output(model, error_type):
    with pytest.raises(error_type):
        generate_tokens(
            model,
            torch.tensor([[1, 2]]),
            device="cpu",
            max_new_tokens=1,
        )


def test_generator_rejects_unsupported_sampling_method():
    with pytest.raises(ValueError, match="Unsupported sampling method"):
        generate_tokens(
            ConstantLogitsModel(next_token_id=33),
            torch.tensor([[1, 2]]),
            device="cpu",
            max_new_tokens=1,
            sampling_method="top_p",
        )

