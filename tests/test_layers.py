"""核心算子层单元测试，包含 RMSNorm 及残差融合等基础算子。"""

import pytest
import torch

from stepinfer.layers.norm import RMSNorm


def test_rmsnorm_forward_shape_and_dtype():
    """验证 RMSNorm 前向计算的输出形状与数据类型保持不变。"""
    hidden_size = 64
    norm = RMSNorm(hidden_size=hidden_size)

    # 2D 输入 [batch, hidden_size]
    x_2d = torch.randn(2, hidden_size, dtype=torch.float32)
    out_2d = norm(x_2d)
    assert out_2d.shape == (2, hidden_size)
    assert out_2d.dtype == torch.float32

    # 3D 输入 [batch, seq_len, hidden_size] 及半精度测试
    x_3d = torch.randn(1, 8, hidden_size, dtype=torch.bfloat16)
    out_3d = norm(x_3d)
    assert out_3d.shape == (1, 8, hidden_size)
    assert out_3d.dtype == torch.bfloat16


def test_rmsnorm_mathematical_correctness():
    """验证 RMSNorm 计算数值与纯数学公式严格对齐。"""
    hidden_size = 4
    eps = 1e-5
    norm = RMSNorm(hidden_size=hidden_size, eps=eps)

    # 固定自定义权重
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    # 手动推导标准 RMS: sqrt(mean(1^2 + 2^2 + 3^2 + 4^2) + eps) = sqrt((30 / 4) + 1e-5)
    expected_rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    expected_out = (x / expected_rms) * norm.weight

    actual_out = norm(x)
    assert torch.allclose(actual_out, expected_out, atol=1e-6)


def test_rmsnorm_fused_residual_add():
    """验证残差融合 (Fused Add + RMSNorm) 的等价性与正确性。"""
    hidden_size = 32
    norm = RMSNorm(hidden_size=hidden_size)

    x = torch.randn(2, 4, hidden_size, dtype=torch.float32)
    residual = torch.randn(2, 4, hidden_size, dtype=torch.float32)

    # 1. 融合前向
    normed_x, updated_residual = norm(x, residual)

    # 2. 分离的理论计算
    expected_residual = x + residual
    expected_normed_x = norm(expected_residual)

    # 校验残差累加值和归一化结果均完全一致
    assert torch.allclose(updated_residual, expected_residual, atol=1e-6)
    assert torch.allclose(normed_x, expected_normed_x, atol=1e-6)


def test_rmsnorm_dimension_and_shape_validation():
    """验证非法输入维度与形状不匹配时的防御性异常拦截。"""
    norm = RMSNorm(hidden_size=16)

    # 隐层维度不匹配
    bad_x = torch.randn(2, 8)
    with pytest.raises(ValueError, match="Expected last dimension to be 16"):
        norm(bad_x)

    # 残差形状不匹配
    good_x = torch.randn(2, 4, 16)
    mismatched_residual = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="Residual shape .* does not match input shape"):
        norm(good_x, mismatched_residual)


def test_rmsnorm_invalid_init_parameters():
    """验证初始化参数合法性校验。"""
    with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
        RMSNorm(hidden_size=0)

    with pytest.raises(ValueError, match="eps must be a positive float"):
        RMSNorm(hidden_size=64, eps=-1e-5)
