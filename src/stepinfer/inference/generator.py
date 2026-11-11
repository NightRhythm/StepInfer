"""实现支持 KV Cache 的自回归 Token 生成状态机。

包含两阶段推理流程：
1. Prefill 阶段：一次性送入全量 Prompt 构建初始 KV 缓存，并采样生成第 1 个 Token；
2. Decode 阶段：每步仅送入最新生成的单 Token 与递增游标位置索引，实现 O(1) 复杂度的极速解码。
"""

from typing import Any

import torch
from torch import Tensor

from stepinfer.cache.kv_cache import KVCache
from stepinfer.inference.sampler import sample_next_token


def generate_tokens(
    model: Any,
    input_ids: Tensor,
    *,
    device: str | torch.device = "cpu",
    max_new_tokens: int = 50,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    padding_mask: Tensor | None = None,
    sampling_method: str = "greedy",
    use_cache: bool = True,
    kv_cache: KVCache | None = None,
) -> tuple[Tensor, int | list[int], str | list[str]]:
    """自回归解码生成 Token ID 序列。

    支持单请求与批量请求、两阶段 KV Cache 高效推理及无 Cache 回退模式。

    参数：
        model: 模型实例 (接收 input_ids, positions, kv_cache 并返回 logits)。
        input_ids: 输入 token IDs，形状为 [batch_size, sequence_length]。
        device: 模型和输入 Tensor 所在设备。
        max_new_tokens: 最多生成多少个新 token。
        eos_token_id: 生成后遇到该 token 时停止；为 None 时不使用 EOS 停止。
        pad_token_id: 填充 token ID；当批量请求中有序列提前结束时，用该 ID 覆盖陪跑 token。
        padding_mask: 对应 input_ids 的有效掩码 [batch_size, sequence_length]，1 为真实 token，0 为 PAD。
        sampling_method: 采样方法，当前支持 "greedy"。
        use_cache: 是否启用 KV Cache 加速推理。
        kv_cache: 可选的预建 KVCache 实例。若为 None 且启用 use_cache，则自动从 model.config 创建。

    返回：
        (generated_ids, token_count, stop_reason)：
        - generated_ids: 包含原始 prompt 和新 token 的完整序列 [batch_size, total_len]；
        - token_count: 单请求返回 int，批量请求返回 list[int]，表示各自实际生成的新 token 数量；
        - stop_reason: 单请求返回 str，批量请求返回 list[str]，表示各自的终止原因描述。

    异常：
        TypeError: input_ids 或模型输出不是 PyTorch Tensor。
        ValueError: 输入形状不合法、max_new_tokens 非正整数、采样方法不支持。
    """
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor")

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be a 2D tensor [batch_size, sequence_length], got {input_ids.ndim}D."
        )

    if input_ids.size(-1) == 0 or input_ids.size(0) == 0:
        raise ValueError("input_ids cannot be empty")

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    device = torch.device(device)
    input_ids = input_ids.to(device)
    batch_size, prompt_len = input_ids.shape

    if padding_mask is not None:
        if not isinstance(padding_mask, Tensor):
            raise TypeError("padding_mask must be a torch.Tensor")
        if padding_mask.shape != input_ids.shape:
            raise ValueError(
                f"padding_mask shape {padding_mask.shape} must match input_ids shape {input_ids.shape}."
            )
        padding_mask = padding_mask.to(device)
    else:
        padding_mask = torch.ones((batch_size, prompt_len), dtype=torch.long, device=device)

    # 判断当前模型与运行环境是否支持 KV Cache
    can_use_cache = use_cache and (kv_cache is not None or hasattr(model, "config"))

    with torch.inference_mode():
        if can_use_cache:
            # -------------------------------------------------------------
            # 模式 A: 基于 KV Cache 的两阶段高效自回归生成 (Prefill + Decode)
            # -------------------------------------------------------------
            if kv_cache is None:
                # 预估最大序列长度，按需要动态创建 KVCache
                max_cache_len = getattr(model.config, "max_position_embeddings", 2048)
                model_dtype = getattr(model, "dtype", None)
                if model_dtype is None and hasattr(model, "parameters"):
                    try:
                        model_dtype = next(model.parameters()).dtype
                    except StopIteration:
                        model_dtype = torch.float32
                else:
                    model_dtype = torch.float32

                kv_cache = KVCache.from_config(
                    config=model.config,
                    max_batch_size=batch_size,
                    max_seq_len=min(max_cache_len, prompt_len + max_new_tokens),
                    dtype=model_dtype,
                    device=device,
                )
            else:
                kv_cache.reset()

            token_counts = [0] * batch_size
            stop_reasons = [""] * batch_size
            is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            generated_ids = input_ids.clone()
            curr_padding_mask = padding_mask.clone()

            # 阶段 1: Prefill (首步 Prompt 批量前向计算)
            positions = (curr_padding_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            logits = model(
                input_ids=generated_ids,
                positions=positions,
                kv_cache=kv_cache,
                padding_mask=curr_padding_mask,
            )
            next_tokens = sample_next_token(logits=logits[:, -1, :], method=sampling_method)

            for i in range(batch_size):
                token_counts[i] += 1
                if eos_token_id is not None and next_tokens[i].item() == eos_token_id:
                    is_finished[i] = True
                    stop_reasons[i] = "Generated EOS token."
                elif token_counts[i] >= max_new_tokens:
                    is_finished[i] = True
                    stop_reasons[i] = f"Reached max_new_tokens limit of {max_new_tokens}."

            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            curr_padding_mask = torch.cat(
                [curr_padding_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
                dim=-1,
            )

            # 阶段 2: Decode (后续单步自回归循环)
            while not is_finished.all():
                # 记录本步之前的完成状态
                prev_finished = is_finished.clone()

                curr_tokens = next_tokens
                if pad_token_id is not None:
                    curr_tokens = torch.where(prev_finished, pad_token_id, curr_tokens)
                curr_input_ids = curr_tokens.unsqueeze(-1)

                # 每个请求根据其有效累计长度计算 2D 绝对位置 [batch_size, 1]
                curr_pos = curr_padding_mask.sum(dim=-1, keepdim=True) - 1

                logits = model(
                    curr_input_ids,
                    positions=curr_pos,
                    kv_cache=kv_cache,
                    padding_mask=curr_padding_mask,
                )
                next_tokens = sample_next_token(logits[:, -1, :], method=sampling_method)

                # 仅对未结束序列累加计数并判断终止
                for i in range(batch_size):
                    if not prev_finished[i]:
                        token_counts[i] += 1
                        if eos_token_id is not None and next_tokens[i].item() == eos_token_id:
                            is_finished[i] = True
                            stop_reasons[i] = "Generated EOS token."
                        elif token_counts[i] >= max_new_tokens:
                            is_finished[i] = True
                            stop_reasons[i] = f"Reached max_new_tokens limit of {max_new_tokens}."

                # 若某序列在之前就已经结束，本步新追加的 token 冻结为 pad_token_id
                tokens_to_append = next_tokens
                if pad_token_id is not None:
                    tokens_to_append = torch.where(prev_finished, pad_token_id, next_tokens)

                generated_ids = torch.cat([generated_ids, tokens_to_append.unsqueeze(-1)], dim=-1)
                curr_padding_mask = torch.cat(
                    [curr_padding_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
                    dim=-1,
                )

            if batch_size == 1:
                return generated_ids, token_counts[0], stop_reasons[0]
            return generated_ids, token_counts, stop_reasons

        else:
            # -------------------------------------------------------------
            # 模式 B: 回退兼容模式 (无 Cache / Mock 模型全量重算)
            # -------------------------------------------------------------
            token_counts = [0] * batch_size
            stop_reasons = [""] * batch_size
            is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            while not is_finished.all():
                prev_finished = is_finished.clone()

                logits = model(input_ids)
                if not isinstance(logits, Tensor):
                    raise TypeError("Model output logits must be a torch.Tensor")
                if (
                    logits.ndim != 3
                    or logits.shape[0] != batch_size
                    or logits.shape[1] == 0
                    or logits.shape[2] == 0
                ):
                    raise ValueError(
                        "Model output logits must be a 3D tensor of shape (batch_size, sequence_length, vocab_size)"
                    )
                next_tokens = sample_next_token(logits[:, -1, :], method=sampling_method)
                for i in range(batch_size):
                    if not prev_finished[i]:
                        token_counts[i] += 1
                        if eos_token_id is not None and next_tokens[i].item() == eos_token_id:
                            is_finished[i] = True
                            stop_reasons[i] = "Generated EOS token."
                        elif token_counts[i] >= max_new_tokens:
                            is_finished[i] = True
                            stop_reasons[i] = f"Reached max_new_tokens limit of {max_new_tokens}."

                tokens_to_append = next_tokens
                if pad_token_id is not None:
                    tokens_to_append = torch.where(prev_finished, pad_token_id, next_tokens)
                input_ids = torch.cat([input_ids, tokens_to_append.unsqueeze(-1)], dim=-1)

            if batch_size == 1:
                return input_ids, token_counts[0], stop_reasons[0]
            return input_ids, token_counts, stop_reasons
