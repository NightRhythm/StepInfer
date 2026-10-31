"""从模型输出的 logits 中选择下一个 token。"""

import torch
from torch import Tensor

SUPPORTED_METHODS = ("greedy",)


def greedy_sample(logits: Tensor) -> Tensor:
    """使用贪心策略选择下一个 token。

    参数：
        logits: 当前步骤的 logits，形状应为 ``[vocab_size]`` 或
            ``[batch_size, vocab_size]``。

    返回：
        下一个 token 的 ID。单条输入返回标量 Tensor，批量输入返回
        形状为 ``[batch_size]`` 的 Tensor。

    异常：
        TypeError: logits 不是 PyTorch Tensor。
        ValueError: logits 不是一维或二维 Tensor。
    """
    if not isinstance(logits, Tensor):
        raise TypeError("Input logits must be a torch.Tensor")

    if logits.dim() not in (1, 2):
        raise ValueError("Input logits must be a 1D or 2D tensor")
    if logits.shape[-1] == 0:
        raise ValueError("logits must contain at least one vocabulary score")
    if logits.ndim == 2 and logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sequence")

    # 一维和二维都适用
    return torch.argmax(logits, dim=-1)


def sample_next_token(
    logits: Tensor,
    *,
    method: str = "greedy",
) -> Tensor:
    """根据采样方法选择下一个 token。

    当前阶段只需要支持 ``greedy``。保留这个统一入口，是为了以后加入
    temperature、top-k 或 top-p 时，不需要修改生成循环的调用方式。

    异常：
        ValueError: method 不是当前支持的采样方法。
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported sampling method: {method}. "
            f"Supported methods: {', '.join(SUPPORTED_METHODS)}."
        )

    return greedy_sample(logits=logits)

