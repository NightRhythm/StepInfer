"""实现基础的自回归 token 生成循环。"""

from typing import Any

import torch
from torch import Tensor

from stepinfer.inference.sampler import sample_next_token


def generate_tokens(
    model: Any,
    input_ids: Tensor,
    *,
    device: str | torch.device,
    max_new_tokens: int = 50,
    eos_token_id: int | None = None,
    sampling_method: str = "greedy",
) -> tuple[Tensor, int, str]:
    """根据输入 token IDs 自回归生成后续 token。

    当前阶段只处理基础生成循环，不处理 KV Cache、流式输出或批量请求。

    参数：
        model: 接收 ``input_ids`` 并返回 logits 的模型。
        input_ids: 单条输入 token IDs，形状必须为 ``[1, sequence_length]``。
        device: 模型和输入 Tensor 所在设备。
        max_new_tokens: 最多生成多少个新 token。
        eos_token_id: 生成后遇到该 token 时停止；为 None 时不使用 EOS 停止。
        sampling_method: 当前支持 ``greedy``。

    返回：
        ``(generated_ids, token_count, stop_reason)``：

        - generated_ids：包含原始 prompt 和新 token 的完整序列；
        - token_count：实际生成的新 token 数量；
        - stop_reason：停止原因。

    异常：
        TypeError: input_ids 不是 PyTorch Tensor。
            模型输出不是 PyTorch Tensor。
        ValueError: input_ids 的形状不符合单条生成要求；
            max_new_tokens 不是正数；模型输出 logits 的维度不正确；
            或 sampling_method 不受支持。

    注意：
        还没有 KV Cache 时，每一轮都把完整序列重新送入模型。这样逻辑
        容易理解，但效率较低；KV Cache 会在后续阶段加入。
    """
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")

    if input_ids.size(-1) == 0:
        raise ValueError("input_ids cannot be empty")

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    input_ids = input_ids.to(device)
    token_count = 0

    with torch.inference_mode():
        while True:
            logits = model(input_ids)
            if not isinstance(logits, Tensor):
                raise TypeError("Model output logits must be a torch.Tensor")
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] == 0 or logits.shape[2] == 0:
                raise ValueError("Model output logits must be a 3D tensor of shape (batch_size, sequence_length, vocab_size)")
            logits = logits[:, -1, :]
            next_token = sample_next_token(logits, method=sampling_method)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            token_count += 1
            if eos_token_id is not None and next_token.item() == eos_token_id:
                stop_reason = "Generated EOS token."
                break
            if token_count >= max_new_tokens:
                stop_reason = f"Reached max_new_tokens limit of {max_new_tokens}."
                break

    
    return input_ids, token_count, stop_reason

