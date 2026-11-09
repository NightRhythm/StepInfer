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
    device: str | torch.device,
    max_new_tokens: int = 50,
    eos_token_id: int | None = None,
    sampling_method: str = "greedy",
    use_cache: bool = True,
    kv_cache: KVCache | None = None,
) -> tuple[Tensor, int, str]:
    """根据输入 token IDs 自回归生成后续 token。

    支持两阶段 KV Cache 推理与无 Cache 回退模式。

    参数：
        model: 模型实例 (接收 input_ids, positions, kv_cache 并返回 logits)。
        input_ids: 单条输入 token IDs，形状必须为 [1, sequence_length]。
        device: 模型和输入 Tensor 所在设备。
        max_new_tokens: 最多生成多少个新 token。
        eos_token_id: 生成后遇到该 token 时停止；为 None 时不使用 EOS 停止。
        sampling_method: 采样方法，当前支持 "greedy"。
        use_cache: 是否启用 KV Cache 加速推理。
        kv_cache: 可选的预建 KVCache 实例。若为 None 且启用 use_cache，则自动从 model.config 创建。

    返回：
        (generated_ids, token_count, stop_reason)：
        - generated_ids: 包含原始 prompt 和新 token 的完整序列 [1, total_len]；
        - token_count: 实际生成的新 token 数量；
        - stop_reason: 终止原因描述。

    异常：
        TypeError: input_ids 或模型输出不是 PyTorch Tensor。
        ValueError: 输入形状不合法、max_new_tokens 非正整数、采样方法不支持。
    """
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor")

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")

    if input_ids.size(-1) == 0:
        raise ValueError("input_ids cannot be empty")

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    device = torch.device(device)
    input_ids = input_ids.to(device)

    # 判断当前模型与运行环境是否支持 KV Cache
    can_use_cache = use_cache and (kv_cache is not None or hasattr(model, "config"))

    with torch.inference_mode():
        if can_use_cache:
            # -------------------------------------------------------------
            # 模式 A: 基于 KV Cache 的两阶段高效自回归生成 (Prefill + Decode)
            # -------------------------------------------------------------
            if kv_cache is None:
                # 预估最大序列长度，按需要动态创建 KVCache
                prompt_len = input_ids.shape[1]
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
                    max_batch_size=1,
                    max_seq_len=min(max_cache_len, prompt_len + max_new_tokens),
                    dtype=model_dtype,
                    device=device,
                )
            else:
                kv_cache.reset()

            token_count = 0
            generated_ids = input_ids.clone()

            # TODO: [用户实现] 阶段 1: Prefill (首步 Prompt 前向计算)
            # 步骤 1: 构造 prompt 对应的绝对位置索引 positions (从 0 到 prompt_len - 1)
            # 步骤 2: 调用 model(input_ids=generated_ids, positions=positions, kv_cache=kv_cache) 得到 logits
            # 步骤 3: 提取最后一个时刻的 logits[:, -1, :]，调用 sample_next_token 采样第 1 个新 Token
            # 步骤 4: 将新 Token 拼接到 generated_ids 末尾，更新 token_count
            # 步骤 5: 检查是否遇到 EOS 或已达到 max_new_tokens 终止
            positions = torch.arange(0, prompt_len, device=device)
            logits = model(input_ids=generated_ids, positions=positions, kv_cache=kv_cache)
            next_token = sample_next_token(logits=logits[:, -1, :], method=sampling_method)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            token_count += 1
            if eos_token_id is not None and next_token.item() == eos_token_id:
                return generated_ids, token_count, "Generated EOS token."
            if token_count >= max_new_tokens:
                return generated_ids, token_count, f"Reached max_new_tokens limit of {max_new_tokens}."

            # TODO: [用户实现] 阶段 2: Decode (后续单步自回归循环)
            # 循环条件: token_count < max_new_tokens
            # 步骤 1: 每次只构造最新单 Token 的输入张量 [1, 1]
            # 步骤 2: 获取当前单步的绝对位置标量 (即当前缓存已有的有效长度 positions = torch.tensor([current_pos]))
            # 步骤 3: 调用 model(input_ids=current_token, positions=positions, kv_cache=kv_cache)
            # 步骤 4: 提取单步 logits[:, -1, :]，采样下一个 Token 并拼接到 generated_ids
            # 步骤 5: 更新 token_count，检查 EOS 或达到最大上限终止
            while token_count < max_new_tokens:
                # 1. 核心关键：每次只送入最新生成的单 Token，形状为 [1, 1]                                                                                                                              
                curr_input_ids = next_token.view(1, 1)
  
                # 2. 当前绝对位置标量：正好等于底层已缓存的历史有效长度 (游标指针)                                                                                                                      
                curr_pos = torch.tensor([kv_cache[0].current_length], device=device)
  
                # 3. 单步前向计算                                                                                                                                                                       
                logits = model(curr_input_ids, positions=curr_pos, kv_cache=kv_cache)
                next_token = sample_next_token(logits[:, -1, :], method=sampling_method)
  
                # 4. 追加结果并推进计数                                                                                                                                                                 
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
                token_count += 1
  
                # 5. 停止条件判断                                                                                                                                                                       
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    return generated_ids, token_count, "Generated EOS token."
  
            return generated_ids, token_count, f"Reached max_new_tokens limit of {max_new_tokens}."

        else:
            # -------------------------------------------------------------
            # 模式 B: 回退兼容模式 (无 Cache / Mock 模型全量重算)
            # -------------------------------------------------------------
            token_count = 0
            while True:
                logits = model(input_ids)
                if not isinstance(logits, Tensor):
                    raise TypeError("Model output logits must be a torch.Tensor")
                if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] == 0 or logits.shape[2] == 0:
                    raise ValueError(
                        "Model output logits must be a 3D tensor of shape (batch_size, sequence_length, vocab_size)"
                    )
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
