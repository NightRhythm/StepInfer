"""通用权重加载器实现，支持 Safetensors 流式读取、Merged GEMM 拼接映射与对齐审计。"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class WeightLoadingResult:
    """权重加载审计结果数据类。

    Attributes:
        total_checkpoint_tensors (int): Checkpoint 文件中包含的权重张量总数。
        loaded_checkpoint_tensors (int): 成功匹配并加载进模型的 Checkpoint 张量总数。
        loaded_parameters_count (int): 成功赋值的模型 Parameter 总数。
        missing_keys (list[str]): 模型中声明但 Checkpoint 中缺失的参数列表。
        unexpected_keys (list[str]): Checkpoint 中存在但模型未使用的参数列表。
        shape_mismatches (list[tuple[str, tuple[int, ...], tuple[int, ...]]]): 形状不匹配项 (key, ckpt_shape, model_shape)。
    """

    total_checkpoint_tensors: int = 0
    loaded_checkpoint_tensors: int = 0
    loaded_parameters_count: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """检查加载是否绝对健康无隐患。"""
        return len(self.missing_keys) == 0 and len(self.shape_mismatches) == 0

    def format_summary(self) -> str:
        """生成清晰可读的结构化权重体检与审计报告字符串。"""
        lines = [
            "=" * 60,
            "StepInfer 权重加载与健康审计报告 (Weight Loading Audit)",
            "=" * 60,
            f"Checkpoint 张量总数:  {self.total_checkpoint_tensors}",
            f"已成功加载张量数:    {self.loaded_checkpoint_tensors} / {self.total_checkpoint_tensors} (100.0%)"
            if self.loaded_checkpoint_tensors == self.total_checkpoint_tensors
            else f"已成功加载张量数:    {self.loaded_checkpoint_tensors} / {self.total_checkpoint_tensors}",
            f"已赋值模型参数项:    {self.loaded_parameters_count}",
            f"缺失参数 (Missing):   {len(self.missing_keys)}",
            f"未预期 (Unexpected):  {len(self.unexpected_keys)}",
            f"形状错位 (Mismatched): {len(self.shape_mismatches)}",
            "-" * 60,
        ]

        if self.missing_keys:
            lines.append("🚨 发现缺失参数 (Missing Keys):")
            for k in self.missing_keys:
                lines.append(f"  - {k}")

        if self.unexpected_keys:
            lines.append("⚠️ 发现未预期参数 (Unexpected Keys):")
            for k in self.unexpected_keys:
                lines.append(f"  - {k}")

        if self.shape_mismatches:
            lines.append("❌ 发现形状不匹配参数 (Shape Mismatches):")
            for k, ckpt_shape, model_shape in self.shape_mismatches:
                lines.append(f"  - {k}: checkpoint {ckpt_shape} vs model {model_shape}")

        if self.is_success:
            lines.append("✅ 结论: 所有参数严丝合缝对齐就绪！")
        else:
            lines.append("❌ 结论: 存在异常参数缺失或错位，请检视上述明细！")

        lines.append("=" * 60)
        return "\n".join(lines)


def _find_safetensors_files(model_path: str | Path) -> list[Path]:
    """定位指定目录或文件路径下的所有 safetensors 权重文件。"""
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Weight path does not exist: {path}")

    if path.is_file():
        if path.suffix == ".safetensors":
            return [path]
        raise ValueError(f"Target file is not a .safetensors file: {path}")

    # 若为目录，检索所有 *.safetensors 文件
    safetensors_files = sorted(list(path.glob("*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in directory: {path}")

    return safetensors_files


def load_model_weights(
    model: nn.Module,
    model_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> WeightLoadingResult:
    """从 Safetensors 文件中流式加载真实预训练权重至模型中。

    核心能力:
        1. 流式低内存加载: 利用 safe_open 逐个读取 Tensor，并在内存中原地拷贝；
        2. Merged GEMM 拼接映射: 将分立的 q/k/v_proj 拼接进 qkv_proj，gate/up_proj 拼接进 gate_up_proj；
        3. 权重对齐审计: 统计并输出 Loaded/Missing/Unexpected/Shape 检查报告。

    参数:
        model (nn.Module): 目标模型实例 (如 Qwen3ForCausalLM)。
        model_path (str | Path): 模型权重所在目录路径或单个 .safetensors 文件路径。
        device (str | torch.device): 权重目标放置设备 (如 'cpu' 或 'cuda')。
        dtype (torch.dtype | None): 目标数据类型 (如 torch.bfloat16 或 torch.float32)，若为 None 则保持权重原始精度。
        strict (bool): 若为 True，当存在非预期的 Missing Keys 或 Shape 错位时主动抛出 RuntimeError。

    返回:
        WeightLoadingResult: 结构化审计结果对象。
    """
    target_device = torch.device(device)
    safetensors_files = _find_safetensors_files(model_path)

    # 抽取模型配置中的动态切片尺寸超参
    config: Any = getattr(model, "config", None)
    if config is not None:
        num_heads = getattr(config, "num_attention_heads", 16)
        num_kv_heads = getattr(config, "num_key_value_heads", 8)
        head_dim = getattr(config, "head_dim", 128)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        intermediate_size = getattr(config, "intermediate_size", 3072)
    else:
        # 默认回退到 Qwen3-0.6B 标准超参
        q_size = 16 * 128
        kv_size = 8 * 128
        intermediate_size = 3072

    # 获取模型所有已注册参数字典与加载跟踪标记
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()

    result = WeightLoadingResult()

    # 遍历所有 safetensors 权重分片
    for file_path in safetensors_files:
        logger.info(f"Loading weights from {file_path.name}...")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                result.total_checkpoint_tensors += 1
                loaded_tensor = f.get_tensor(weight_name)

                if dtype is not None:
                    loaded_tensor = loaded_tensor.to(dtype=dtype)

                # ==========================================================
                # 映射规则 1: QKV 融合投影 (q_proj, k_proj, v_proj -> qkv_proj)
                # ==========================================================
                if "self_attn.q_proj.weight" in weight_name:
                    param_name = weight_name.replace("self_attn.q_proj.weight", "self_attn.qkv_proj.weight")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        param.data.narrow(0, 0, q_size).copy_(loaded_tensor)
                        loaded_params.add(param_name)
                        result.loaded_checkpoint_tensors += 1
                    else:
                        result.unexpected_keys.append(weight_name)

                elif "self_attn.k_proj.weight" in weight_name:
                    param_name = weight_name.replace("self_attn.k_proj.weight", "self_attn.qkv_proj.weight")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        param.data.narrow(0, q_size, kv_size).copy_(loaded_tensor)
                        loaded_params.add(param_name)
                        result.loaded_checkpoint_tensors += 1
                    else:
                        result.unexpected_keys.append(weight_name)

                elif "self_attn.v_proj.weight" in weight_name:
                    param_name = weight_name.replace("self_attn.v_proj.weight", "self_attn.qkv_proj.weight")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        param.data.narrow(0, q_size + kv_size, kv_size).copy_(loaded_tensor)
                        loaded_params.add(param_name)
                        result.loaded_checkpoint_tensors += 1
                    else:
                        result.unexpected_keys.append(weight_name)

                # ==========================================================
                # 映射规则 2: SwiGLU 门控前馈融合 (gate_proj, up_proj -> gate_up_proj)
                # ==========================================================
                elif "mlp.gate_proj.weight" in weight_name:
                    param_name = weight_name.replace("mlp.gate_proj.weight", "mlp.gate_up_proj.weight")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        param.data.narrow(0, 0, intermediate_size).copy_(loaded_tensor)
                        loaded_params.add(param_name)
                        result.loaded_checkpoint_tensors += 1
                    else:
                        result.unexpected_keys.append(weight_name)

                elif "mlp.up_proj.weight" in weight_name:
                    param_name = weight_name.replace("mlp.up_proj.weight", "mlp.gate_up_proj.weight")
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        param.data.narrow(0, intermediate_size, intermediate_size).copy_(loaded_tensor)
                        loaded_params.add(param_name)
                        result.loaded_checkpoint_tensors += 1
                    else:
                        result.unexpected_keys.append(weight_name)

                # ==========================================================
                # 映射规则 3: 直连参数 (Embedding, Norm, Down Proj, O Proj)
                # ==========================================================
                elif weight_name in params_dict:
                    param = params_dict[weight_name]
                    if param.shape != loaded_tensor.shape:
                        result.shape_mismatches.append(
                            (weight_name, tuple(loaded_tensor.shape), tuple(param.shape))
                        )
                    else:
                        param.data.copy_(loaded_tensor)
                        loaded_params.add(weight_name)
                        result.loaded_checkpoint_tensors += 1
                else:
                    result.unexpected_keys.append(weight_name)

    # 权重共享处理 (Weight Tying):
    # 若模型开启了 tie_word_embeddings，lm_head.weight 共享 embed_tokens.weight 的数据内存指针
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    if tie_word_embeddings and "model.embed_tokens.weight" in loaded_params:
        loaded_params.add("lm_head.weight")

    # 审计 Missing Keys
    for param_name in params_dict.keys():
        if param_name not in loaded_params:
            result.missing_keys.append(param_name)

    result.loaded_parameters_count = len(loaded_params)

    # 统一设备迁移与评估模式设置
    model.to(device=target_device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()

    # 打印审计体检报告
    summary = result.format_summary()
    logger.info("\n" + summary)

    # 严格模式校验
    if strict and not result.is_success:
        error_msg = f"Weight loading audit failed:\n{summary}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return result
