"""核心算子层单元测试，包含 RMSNorm 及残差融合等基础算子。"""

import pytest
import torch
import torch.nn.functional as F

from stepinfer.layers.activation import SiluAndMul
from stepinfer.layers.norm import RMSNorm
from stepinfer.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


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


# ==============================================================================
# RoPE (旋转位置编码) 测试用例
# ==============================================================================


def test_apply_rotary_emb_identity_at_zero():
    """验证当角度为 0 (cos=1, sin=0) 时，输入保持严格不变。"""
    x = torch.randn(2, 4, 8, 64, dtype=torch.float32)
    cos = torch.ones(2, 4, 1, 32, dtype=torch.float32)
    sin = torch.zeros(2, 4, 1, 32, dtype=torch.float32)

    out = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-6)
    assert out.dtype == torch.float32


def test_apply_rotary_emb_half_split_mathematical_rotation():
    """验证 Half-Split 旋转在 90 度 (cos=0, sin=1) 时的几何变换: [x1, x2] -> [-x2, x1]。"""
    # 构造确定性向量: 前半段为 [1.0, 2.0]，后半段为 [3.0, 4.0]
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)
    cos = torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float32)
    sin = torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float32)

    # 预期: y1 = x1 * 0 - x2 * 1 = [-3.0, -4.0]
    #       y2 = x2 * 0 + x1 * 1 = [1.0, 2.0]
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]], dtype=torch.float32)

    out = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rotary_embedding_relative_distance_invariance():
    """验证 RoPE 核心定理: 相对距离相等时，Query 与 Key 的点积内积完全相等。

    即: <R_m(q), R_n(k)> 仅取决于 (m - n)。
    """
    head_dim = 64
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # 情况 A: m=5, n=2 (相对距离 = 3)
    q_pos_a = torch.tensor([5])
    k_pos_a = torch.tensor([2])
    q_rot_a, _ = rope(q_pos_a, q, q)
    _, k_rot_a = rope(k_pos_a, k, k)
    dot_product_a = torch.sum(q_rot_a * k_rot_a)

    # 情况 B: m=20, n=17 (相对距离 = 3)
    q_pos_b = torch.tensor([20])
    k_pos_b = torch.tensor([17])
    q_rot_b, _ = rope(q_pos_b, q, q)
    _, k_rot_b = rope(k_pos_b, k, k)
    dot_product_b = torch.sum(q_rot_b * k_rot_b)

    # 无论绝对位置相差多大，只要相对距离为 3，旋转后的点积必须高度一致
    assert torch.allclose(dot_product_a, dot_product_b, atol=1e-5)


def test_rotary_embedding_1d_and_2d_positions_with_gqa_heads():
    """验证 1D 与 2D positions，以及 GQA 头数不一致 (如 Q:16, KV:8) 时的广播正确性。"""
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    seq_len = 5
    batch_size = 2

    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=1024)

    # 1. 测试 1D positions: [seq_len]
    pos_1d = torch.arange(seq_len)
    q_1d = torch.randn(1, seq_len, num_q_heads, head_dim, dtype=torch.bfloat16)
    k_1d = torch.randn(1, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)
    q_rot_1d, k_rot_1d = rope(pos_1d, q_1d, k_1d)

    assert q_rot_1d.shape == (1, seq_len, num_q_heads, head_dim)
    assert k_rot_1d.shape == (1, seq_len, num_kv_heads, head_dim)
    assert q_rot_1d.dtype == torch.bfloat16
    assert k_rot_1d.dtype == torch.bfloat16

    # 2. 测试 2D positions: [batch_size, seq_len]
    pos_2d = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    q_2d = torch.randn(batch_size, seq_len, num_q_heads, head_dim)
    k_2d = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
    q_rot_2d, k_rot_2d = rope(pos_2d, q_2d, k_2d)

    assert q_rot_2d.shape == (batch_size, seq_len, num_q_heads, head_dim)
    assert k_rot_2d.shape == (batch_size, seq_len, num_kv_heads, head_dim)


def test_rotary_embedding_validation_and_errors():
    """验证 RoPE 参数初始化与位置超界的防御性异常拦截。"""
    # 非法 head_dim (必须为正偶数)
    with pytest.raises(ValueError, match="head_dim must be a positive even integer"):
        RotaryEmbedding(head_dim=63)
    with pytest.raises(ValueError, match="head_dim must be a positive even integer"):
        RotaryEmbedding(head_dim=-64)

    # 位置越界检查
    rope = RotaryEmbedding(head_dim=32, max_position_embeddings=100)
    q = torch.randn(1, 2, 4, 32)
    k = torch.randn(1, 2, 2, 32)

    # 负数位置
    with pytest.raises(ValueError, match="Position indices out of bounds"):
        rope(torch.tensor([-1, 0]), q, k)

    # 超过 max_position_embeddings
    with pytest.raises(ValueError, match="Position indices out of bounds"):
        rope(torch.tensor([0, 100]), q, k)

    # 非法 3D positions
    with pytest.raises(ValueError, match="positions tensor must be 1D or 2D"):
        rope(torch.zeros(1, 2, 3, dtype=torch.long), q, k)


# ==============================================================================
# SiluAndMul (SwiGLU 激活算子) 测试用例
# ==============================================================================


def test_silu_and_mul_forward_shape_and_dtype():
    """验证 SiluAndMul 算子的输出形状（减半）与数据类型保持不变。"""
    act = SiluAndMul()
    intermediate_size = 128

    # 1. 2D 输入 [batch, 2 * intermediate_size]
    x_2d = torch.randn(2, 2 * intermediate_size, dtype=torch.float32)
    out_2d = act(x_2d)
    assert out_2d.shape == (2, intermediate_size)
    assert out_2d.dtype == torch.float32

    # 2. 3D 输入 [batch, seq_len, 2 * intermediate_size] 及 bfloat16 支持
    x_3d = torch.randn(2, 8, 2 * intermediate_size, dtype=torch.bfloat16)
    out_3d = act(x_3d)
    assert out_3d.shape == (2, 8, intermediate_size)
    assert out_3d.dtype == torch.bfloat16


def test_silu_and_mul_mathematical_correctness():
    """验证 SiluAndMul 计算数值与理论公式 SiLU(gate) * up 严格对齐。"""
    act = SiluAndMul()

    # 构造确定性张量
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
    # 前半段 gate: [1.0, -2.0], 后半段 up: [3.0, 0.5]
    gate, up = torch.chunk(x, chunks=2, dim=-1)
    expected = F.silu(gate) * up

    actual = act(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_silu_and_mul_odd_dimension_raises_error():
    """验证输入张量末尾维度为奇数时抛出 ValueError。"""
    act = SiluAndMul()
    bad_x = torch.randn(2, 4, 15)  # 15 不能被 2 整除

    with pytest.raises(ValueError, match="The last dimension of input tensor must be even"):
        act(bad_x)
