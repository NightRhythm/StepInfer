import pytest
import torch

from stepinfer.inference.sampler import greedy_sample, sample_next_token


def test_greedy_sample_selects_highest_logit_for_one_sequence():
    logits = torch.tensor([0.1, 0.7, 0.2])

    token_id = greedy_sample(logits)

    assert token_id.ndim == 0
    assert token_id.item() == 1


def test_greedy_sample_selects_highest_logit_for_each_batch_item():
    logits = torch.tensor([[0.1, 0.7, 0.2], [0.9, 0.1, 0.3]])

    token_ids = greedy_sample(logits)

    assert torch.equal(token_ids, torch.tensor([1, 0]))


def test_greedy_sample_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="torch.Tensor"):
        greedy_sample([0.1, 0.9])


@pytest.mark.parametrize(
    "logits",
    [torch.zeros(1, 1, 3), torch.zeros(0, 3), torch.zeros(2, 0)],
)
def test_greedy_sample_rejects_invalid_or_empty_shapes(logits):
    with pytest.raises(ValueError):
        greedy_sample(logits)


def test_sample_next_token_rejects_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported sampling method"):
        sample_next_token(torch.tensor([0.1, 0.9]), method="top_k")

