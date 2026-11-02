"""SwiGLU 门控前馈网络所需的激活函数算子实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiluAndMul 激活算子 (SwiGLU 门控前馈网络专用)。

    对合并投影 (Merged gate_up_proj) 产生的输出张量进行切半与门控相乘。
    相比于单独做 gate_proj 与 up_proj，合并投影配合本算子可减少一次 GPU 矩阵乘法核心发射。

    数学公式:
        输入 x 形状为 [..., 2 * d]，在最后一个维度均分为 x1 与 x2:
        output = SiLU(x1) * x2
               = (x1 * sigmoid(x1)) * x2

    输入与输出维度 (Tensor Shapes):
        输入 x: [..., 2 * intermediate_size]
        输出:   [..., intermediate_size]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算。

        参数:
            x (torch.Tensor): 形状为 [..., 2 * intermediate_size] 的张量。

        返回:
            torch.Tensor: 形状为 [..., intermediate_size] 的张量。
        """
        if x.size(-1) % 2 != 0:
            raise ValueError(
                f"The last dimension of input tensor must be even for SiluAndMul, but got {x.size(-1)}."
            )

        gate, up = torch.chunk(x, chunks=2, dim=-1)
        return F.silu(gate) * up
