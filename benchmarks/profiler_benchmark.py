"""模型推理算子耗时剖析基准测试。

使用 PyTorch Profiler 追踪单次推理的计算图执行、算子调用耗时与显存事件，
导出 Chrome Trace 时间线（支持 Perfetto / Chrome DevTools 可视化）及 Markdown 算子明细。
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

# 允许在未安装项目包的情况下，从仓库根目录直接运行。
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch

from stepinfer.cache.kv_cache import KVCache
from stepinfer.config.model_config import ModelConfig
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.sampler import sample_next_token
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer
from stepinfer.utils.profiler import TorchProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _run_generation(
    model: Qwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    use_cache: bool = True,
) -> torch.Tensor:
    input_ids = prompt_ids
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    kv_cache = None
    if use_cache:
        model_dtype = next(model.parameters()).dtype
        kv_cache = KVCache.from_config(
            config=model.config,
            max_batch_size=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
            dtype=model_dtype,
            device=device,
        )

    with torch.inference_mode():
        if use_cache and kv_cache is not None:
            # 1. Prefill 阶段
            positions = torch.arange(prompt_len, device=device)
            logits = model(input_ids, positions=positions, kv_cache=kv_cache)
            next_token = sample_next_token(logits[:, -1, :], method="greedy")
            input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)

            # 2. Decode 阶段
            for _ in range(max_new_tokens - 1):
                curr_pos = torch.full((batch_size, 1), kv_cache[0].current_length, device=device)
                logits = model(next_token.unsqueeze(-1), positions=curr_pos, kv_cache=kv_cache)
                next_token = sample_next_token(logits[:, -1, :], method="greedy")
                input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
        else:
            for _ in range(max_new_tokens):
                logits = model(input_ids)
                next_token = sample_next_token(logits[:, -1, :], method="greedy")
                input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)

    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize(prompt_ids.device)
    return input_ids


def _make_prompt_ids(
    length: int,
    vocab_size: int,
    *,
    batch_size: int = 1,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        low=100,
        high=vocab_size - 100,
        size=(batch_size, length),
        generator=generator,
        device="cpu",
        dtype=torch.long,
    ).to(device)


def _derive_model_tag(model_path: Path) -> str:
    name = model_path.name.lower()
    if "0.6b" in name:
        return "0.6b"
    if "1.7b" in name:
        return "1.7b"
    return name.replace("-", "_")


def _format_size(num_bytes: int) -> str:
    abs_bytes = abs(num_bytes)
    sign = "-" if num_bytes < 0 else ""
    if abs_bytes >= 1024 * 1024 * 1024:
        return f"{sign}{abs_bytes / (1024 ** 3):.2f} GB"
    if abs_bytes >= 1024 * 1024:
        return f"{sign}{abs_bytes / (1024 ** 2):.2f} MB"
    if abs_bytes >= 1024:
        return f"{sign}{abs_bytes / 1024:.2f} KB"
    return f"{sign}{abs_bytes} B"


def _render_profiler_report(
    *,
    model_name: str,
    num_params: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt_length: int,
    max_new_tokens: int,
    profiler: TorchProfiler,
    batch_size: int = 1,
    trace_rel_path: str | None = None,
    top_n: int = 15,
    use_cache: bool = True,
) -> str:
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    dtype_str = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")
    cache_str = "Static pre-allocated (in-place slicing)" if use_cache else "None"
    trace_display = f"[`{trace_rel_path}`]({trace_rel_path})" if trace_rel_path else "None (disabled)"

    lines = [
        f"# Profiler Breakdown — {model_name}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|:---|:---|",
        f"| Model | {model_name} |",
        f"| Parameters | {num_params:,} |",
        f"| Precision | {dtype_str} |",
        f"| Device | {device_name} |",
        f"| PyTorch | {torch.__version__} |",
        f"| Python | {platform.python_version()} |",
        f"| KV Cache | {cache_str} |",
        f"| Batch Size | {batch_size} |",
        f"| Prompt Tokens | {prompt_length} |",
        f"| Generated Tokens | {max_new_tokens} |",
        f"| Chrome Trace | {trace_display} |",
        "",
        "---",
        "",
    ]

    events = profiler._prof.key_averages()  # type: ignore[union-attr]

    # 判断是否有实际采集到的 CUDA kernel 活动
    has_cuda_events = any(getattr(e, "device_time_total", 0) > 0 for e in events)

    if has_cuda_events:
        sorted_events = sorted(events, key=lambda e: getattr(e, "device_time_total", 0), reverse=True)
        total_time = sum(getattr(e, "self_device_time_total", 0) for e in sorted_events)

        lines.extend([
            f"## Top {min(top_n, len(sorted_events))} Operations (Sorted by CUDA Total)",
            "",
            "| Op | CUDA Total | Self CUDA % | CUDA Time Avg | # Calls | CUDA Mem |",
            "|:---|---:|---:|---:|---:|---:|",
        ])
        for e in sorted_events[:top_n]:
            self_time = getattr(e, "self_device_time_total", 0)
            self_pct = (self_time / total_time * 100.0) if total_time > 0 else 0.0
            avg_time = getattr(e, "device_time_str", "N/A")
            mem_str = _format_size(getattr(e, "device_memory_usage", 0))
            lines.append(
                f"| `{e.key}` | {e.device_time_total_str} | {self_pct:.1f}% | {avg_time} | {e.count} | {mem_str} |"
            )
    else:
        sorted_events = sorted(events, key=lambda e: getattr(e, "cpu_time_total", 0), reverse=True)
        total_time = sum(getattr(e, "self_cpu_time_total", 0) for e in sorted_events)

        lines.extend([
            f"## Top {min(top_n, len(sorted_events))} Operations (Sorted by CPU Total / Dispatch)",
            "",
            "| Op | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem |",
            "|:---|---:|---:|---:|---:|---:|",
        ])
        for e in sorted_events[:top_n]:
            self_time = getattr(e, "self_cpu_time_total", 0)
            self_pct = (self_time / total_time * 100.0) if total_time > 0 else 0.0
            avg_time = getattr(e, "cpu_time_str", "N/A")
            mem_str = _format_size(getattr(e, "device_memory_usage", 0))
            lines.append(
                f"| `{e.key}` | {e.cpu_time_total_str} | {self_pct:.1f}% | {avg_time} | {e.count} | {mem_str} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Profiler Detailed Table",
        "",
        "```text",
        profiler.summary(row_limit=top_n),
        "```",
        "",
    ])

    return "\n".join(lines)


def run_profiler_benchmark(
    model_path: Path,
    device: str = "cuda:0",
    batch_size: int = 1,
    prompt_length: int = 512,
    max_new_tokens: int = 1,
    warmup_runs: int = 2,
    seed: int = 42,
    tag: str | None = None,
    raw_dir: Path = Path("docs/benchmarks/raw"),
    use_cache: bool = True,
    save_trace: bool = True,
) -> tuple[Path | None, Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")

    model_path = model_path.resolve()
    model_tag = tag if tag is not None else _derive_model_tag(model_path)

    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = target_device.type == "cuda" and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    logger.info("Loading Qwen3 model from %s", model_path)
    config = Qwen3Config.from_json_file(model_path / "config.json")
    model = Qwen3ForCausalLM(config=config)
    load_model_weights(model=model, model_path=model_path, device=device, dtype=dtype)
    model.eval()
    tokenizer = HFTokenizer(ModelConfig(name=model_path.name, model_dir=model_path))

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model ready: %d parameters, device=%s, dtype=%s, batch_size=%d, use_cache=%s",
        num_params,
        target_device,
        dtype,
        batch_size,
        use_cache,
    )

    prompt_ids = _make_prompt_ids(
        length=prompt_length,
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        seed=seed,
        device=target_device,
    )

    for i in range(warmup_runs):
        logger.info("Warmup run %d/%d (not profiled)...", i + 1, warmup_runs)
        out = _run_generation(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        tokenizer.decode(out[0].tolist())
        del out

    logger.info(
        "Starting Profiler for %s (batch_size=%d, prompt_length=%d, max_new_tokens=%d, use_cache=%s)...",
        model_path.name,
        batch_size,
        prompt_length,
        max_new_tokens,
        use_cache,
    )

    profiler = TorchProfiler(use_cuda=use_cuda, record_shapes=True, profile_memory=True)
    with profiler:
        out = _run_generation(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        tokenizer.decode(out[0].tolist())
        del out

    resolved_raw_dir = raw_dir if raw_dir.is_absolute() else _repo_root / raw_dir
    md_dir = resolved_raw_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    if batch_size > 1:
        md_prefix = "batch"
    elif use_cache:
        md_prefix = "kvcache"
    else:
        md_prefix = "baseline"

    md_name = f"{md_prefix}_profiler_{model_tag}.md" if not model_tag.startswith(md_prefix) else f"profiler_{model_tag}.md"
    md_path = md_dir / md_name

    trace_path: Path | None = None
    trace_rel_for_md: str | None = None
    if save_trace:
        traces_dir = resolved_raw_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_name = f"{md_prefix}_trace_{model_tag}_prompt{prompt_length}.json" if not model_tag.startswith(md_prefix) else f"trace_{model_tag}_prompt{prompt_length}.json"
        trace_path = traces_dir / trace_name
        profiler.export_chrome_trace(trace_path)
        logger.info("Exported Chrome trace to %s", trace_path)
        try:
            trace_rel_for_md = f"../traces/{trace_path.name}"
        except ValueError:
            trace_rel_for_md = str(trace_path)

    report = _render_profiler_report(
        model_name=model_path.name,
        num_params=num_params,
        device=target_device,
        dtype=dtype,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        profiler=profiler,
        batch_size=batch_size,
        trace_rel_path=trace_rel_for_md,
        top_n=15,
        use_cache=use_cache,
    )
    md_path.write_text(report, encoding="utf-8")
    logger.info("Saved profiler markdown to %s", md_path)
    print("\n" + report)
    return trace_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="StepInfer single/batch request profiler benchmark")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("assets/models/Qwen3-0.6B-Base"),
        help="Path to pretrained model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to profile",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Synthetic prompt token length to profile",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Fixed number of generated tokens (default: 1 for single forward prefill profiling)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs prior to profiling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default=None, help="Model tag (e.g. 0.6b, 1.7b)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("docs/benchmarks/raw"),
        help="Raw benchmark artifacts root directory",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV Cache to profile baseline recalculation",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip exporting Chrome trace json file",
    )

    args = parser.parse_args()
    run_profiler_benchmark(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        tag=args.tag,
        raw_dir=args.raw_dir,
        use_cache=not args.no_cache,
        save_trace=not args.no_trace,
    )


if __name__ == "__main__":
    main()
