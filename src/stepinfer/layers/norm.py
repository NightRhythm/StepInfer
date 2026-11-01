"""均方根归一化 (RMSNorm) 及残差融合 (Fused Add + RMSNorm) 算子实现。"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) 算子。

    相比传统 LayerNorm，去除了均值计算 (Mean-centering)，仅保留方差缩放，
    在保持数值稳定性的同时显著降低了计算量。同时支持残差融合计算 (Fused Add + RMSNorm)。

    数学公式:
        y = (x / RMS(x)) * weight
        其中 RMS(x) = sqrt(mean(x^2) + eps) = rsqrt(mean(x^2) + eps) 的倒数

    Attributes:
        hidden_size (int): 隐层特征维度大小。
        eps (float): 防止除以零的微小值 epsilon (默认 1e-6)。
        weight (nn.Parameter): 可学习的缩放权重向量，形状为 [hidden_size]，初始化为全 1。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be a positive float, got {eps}")

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 Tensor 进行基础 RMSNorm 归一化计算。

        为了保证低精度 (float16 / bfloat16) 下平方累加不发生溢出或下溢，
        先在 float32 下统计方差并计算缩放比例，随后转回原始精度并乘以可学习权重。
        """
        orig_dtype = x.dtype
        x_fp32 = x.float()

        # 计算末尾维度的均方根方差: mean(x^2, dim=-1, keepdim=True)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)

        return (x_normed * self.weight).to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播计算，支持独立归一化与残差融合 (Fused Add + RMSNorm)。

        输入维度变化 (Tensor Shapes):
            x: [..., hidden_size]
            residual (可选): [..., hidden_size]，必须与 x 形状完全一致

        返回:
            若 residual is None:
                返回 normed_x: [..., hidden_size]
            若 residual is not None:
                返回 (normed_x, updated_residual): 两个张量形状均为 [..., hidden_size]
        """
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected last dimension to be {self.hidden_size}, got {x.size(-1)}"
            )

        # 残差融合分支 (Fused Add + RMSNorm)
        if residual is not None:
            if residual.shape != x.shape:
                raise ValueError(
                    f"Residual shape {residual.shape} does not match input shape {x.shape}"
                )
            residual = residual + x
            normed_x = self._norm(residual)
            return normed_x, residual

        # 独立归一化分支 (标准 RMSNorm)
        return self._norm(x)
