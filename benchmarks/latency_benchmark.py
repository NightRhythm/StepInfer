"""单请求生成延迟与吞吐量基准测试。

专注于测量推理延迟、生成吞吐量、显存开销及 MFU，不包含 Profiler 开销。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# 允许在未安装项目包的情况下，从仓库根目录直接运行。
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch

from benchmarks.metrics import BenchmarkResult, BenchmarkTracker, aggregate_results, get_peak_tflops
from stepinfer.cache.kv_cache import KVCache
from stepinfer.config.model_config import ModelConfig
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.sampler import sample_next_token
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_LENGTHS = [32, 64, 256, 512, 1024]


def _run_generation(
    model: Qwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    tracker: BenchmarkTracker | None = None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, BenchmarkResult | None]:
    """从精确的 token ID 执行固定长度贪心解码。

    分词、主机到设备的数据传输和文本解码不计入本模型生成基准。
    关闭 EOS 停止，使每次实验执行相同数量的生成步骤。
    """
    input_ids = prompt_ids
    prompt_len = prompt_ids.shape[-1]
    device = prompt_ids.device

    kv_cache = None
    if use_cache:
        model_dtype = next(model.parameters()).dtype
        kv_cache = KVCache.from_config(
            config=model.config,
            max_batch_size=1,
            max_seq_len=prompt_len + max_new_tokens,
            dtype=model_dtype,
            device=device,
        )

    if tracker is not None:
        tracker.start()

    with torch.inference_mode():
        if use_cache and kv_cache is not None:
            # 1. Prefill 阶段
            positions = torch.arange(prompt_len, device=device)
            logits = model(input_ids, positions=positions, kv_cache=kv_cache)
            next_token = sample_next_token(logits[:, -1, :], method="greedy")
            input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
            if tracker is not None:
                tracker.record_step()

            # 2. Decode 阶段
            for _ in range(max_new_tokens - 1):
                curr_pos = torch.tensor([kv_cache[0].current_length], device=device)
                logits = model(next_token.view(1, 1), positions=curr_pos, kv_cache=kv_cache)
                next_token = sample_next_token(logits[:, -1, :], method="greedy")
                input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
                if tracker is not None:
                    tracker.record_step()
        else:
            for _ in range(max_new_tokens):
                logits = model(input_ids)
                next_token = sample_next_token(logits[:, -1, :], method="greedy")
                input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
                if tracker is not None:
                    tracker.record_step()

    if tracker is None:
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize(prompt_ids.device)
        return input_ids, None
    return input_ids, tracker.finish(prompt_tokens=prompt_ids.shape[-1])


def _make_prompt_ids(
    length: int,
    vocab_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if vocab_size <= 200:
        raise ValueError("vocab_size must be greater than 200 for synthetic token generation")
    token_ids = torch.randint(
        low=100,
        high=vocab_size - 100,
        size=(1, length),
        generator=generator,
        device="cpu",
        dtype=torch.long,
    )
    return token_ids.to(device)


def _range_text(values: list[float], precision: int = 3) -> str:
    med = statistics.median(values)
    vmin = min(values)
    vmax = max(values)
    if precision == 2:
        return f"{med:.2f} [{vmin:.2f}, {vmax:.2f}]"
    return f"{med:.3f} [{vmin:.3f}, {vmax:.3f}]"


def _render_report(
    *,
    model_name: str,
    num_params: int,
    device: torch.device,
    dtype: torch.dtype,
    peak_tflops: float | None,
    prompt_lengths: list[int],
    runs_by_length: dict[int, list[BenchmarkResult]],
    use_cache: bool = True,
) -> str:
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    dtype_str = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")
    peak_tflops_str = f"{peak_tflops:.1f}" if peak_tflops is not None else "N/A"
    cache_str = "Static pre-allocated (in-place slicing)" if use_cache else "None"

    lines = [
        f"# Latency Benchmark — {model_name}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|:---|:---|",
        f"| Model | {model_name} |",
        f"| Parameters | {num_params:,} |",
        f"| Precision | {dtype_str} |",
        f"| Device | {device_name} |",
        f"| Peak TFLOPS ({dtype_str}) | {peak_tflops_str} |",
        f"| PyTorch | {torch.__version__} |",
        f"| Python | {platform.python_version()} |",
        f"| KV Cache | {cache_str} |",
        "| Batching | Single request |",
        "| Sampling | Greedy (argmax) |",
        "",
        "---",
        "",
        "## Results",
        "",
        "数值格式为 `中位数 [最小值, 最大值]`。",
        "",
        "| Prompt Len | Output Tok | Latency (s) | Tok/s | Avg ITL (ms) | P50 ITL (ms) | P90 ITL (ms) | Peak Mem (MB) | Delta Mem (MB) | MFU (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for length in prompt_lengths:
        rows = runs_by_length[length]

        def range_for(name: str, precision: int = 3) -> str:
            values = [getattr(row, name) for row in rows]
            present = [float(value) for value in values if value is not None]
            return _range_text(present, precision=precision) if present else "N/A"

        output_tokens = str(rows[0].generated_tokens)
        latency_str = range_for("generation_wall_time_sec", 3)
        throughput_str = range_for("generation_wall_throughput_tok_per_sec", 2)
        avg_itl_str = range_for("avg_itl_ms", 2)
        p50_str = range_for("p50_itl_ms", 2)
        p90_str = range_for("p90_itl_ms", 2)
        peak_mem_str = range_for("peak_allocated_mb", 2)
        delta_mem_str = range_for("peak_allocated_delta_mb", 2)
        mfu_str = range_for("mfu_percent", 2)

        lines.append(
            f"| {length} | {output_tokens} | {latency_str} | {throughput_str} | "
            f"{avg_itl_str} | {p50_str} | {p90_str} | {peak_mem_str} | {delta_mem_str} | {mfu_str} |"
        )

    return "\n".join(lines) + "\n"


def _derive_model_tag(model_path: Path) -> str:
    name = model_path.name.lower()
    if "0.6b" in name:
        return "0.6b"
    if "1.7b" in name:
        return "1.7b"
    return name.replace("-", "_")


def run_latency_benchmark(
    model_path: Path,
    device: str,
    prompt_lengths: list[int],
    max_new_tokens: int = 32,
    warmup_runs: int = 3,
    repeats: int = 10,
    seed: int = 42,
    tag: str | None = None,
    raw_dir: Path = Path("docs/benchmarks/raw"),
    use_cache: bool = True,
    save_json: bool = True,
) -> tuple[Path, Path | None]:
    if not prompt_lengths or any(length <= 0 for length in prompt_lengths):
        raise ValueError("prompt_lengths must contain positive lengths")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_runs <= 0:
        raise ValueError("warmup_runs must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if len(set(prompt_lengths)) != len(prompt_lengths):
        raise ValueError("prompt_lengths must not contain duplicates")

    model_path = model_path.resolve()
    try:
        display_model_path = model_path.relative_to(_repo_root)
    except ValueError:
        display_model_path = model_path

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

    logger.info("Loading Qwen3 model from %s", display_model_path)
    config = Qwen3Config.from_json_file(model_path / "config.json")
    model = Qwen3ForCausalLM(config=config)
    load_model_weights(model=model, model_path=model_path, device=device, dtype=dtype)
    model.eval()
    tokenizer = HFTokenizer(ModelConfig(name=model_path.name, model_dir=model_path))

    num_params = sum(p.numel() for p in model.parameters())
    peak_tflops = get_peak_tflops(target_device, dtype) if use_cuda else None

    logger.info(
        "Model ready: %d parameters, device=%s, dtype=%s, peak_tflops=%s, use_cache=%s",
        num_params,
        target_device,
        dtype,
        peak_tflops,
        use_cache,
    )

    torch.manual_seed(seed)
    runs_by_length: dict[int, list[BenchmarkResult]] = {}
    prompts_by_length: dict[int, list[int]] = {}

    def git_output(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_repo_root, text=True,
                stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git_output("status", "--porcelain")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_output("rev-parse", "HEAD"),
        "worktree_dirty": None if status is None else bool(status),
        "model_name": model_path.name,
        "model_path": str(display_model_path),
        "num_params": num_params,
        "device": str(target_device),
        "dtype": str(dtype),
        "peak_tflops": peak_tflops,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "seed": seed,
        "prompt_lengths": prompt_lengths,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "batch_size": 1,
        "kv_cache": "static_preallocated" if use_cache else "none",
        "ignore_eos": True,
        "scope": "resident input IDs -> fixed-length generation; excludes text encode/decode",
        "timing_backend": "cuda_event" if use_cuda else "cpu_perf_counter",
    }
    if use_cuda:
        props = torch.cuda.get_device_properties(target_device)
        metadata.update(gpu_name=props.name, gpu_total_bytes=props.total_memory)

    for length in prompt_lengths:
        prompt_ids = _make_prompt_ids(
            length,
            config.vocab_size,
            generator=torch.Generator(device="cpu").manual_seed(seed + length),
            device=target_device,
        )
        prompts_by_length[length] = prompt_ids[0].tolist()
        logger.info("Prompt length %d: %d warmup runs, %d measured runs", length, warmup_runs, repeats)

        for warmup_index in range(warmup_runs):
            logger.info("Warmup %d/%d for prompt length %d", warmup_index + 1, warmup_runs, length)
            warmup_ids, _ = _run_generation(model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
            tokenizer.decode(warmup_ids.squeeze(0).tolist())
            del warmup_ids

        rows: list[BenchmarkResult] = []
        for run_index in range(repeats):
            tracker = BenchmarkTracker(
                device=target_device,
                num_params=num_params,
                dtype=dtype,
                peak_tflops=peak_tflops,
            )
            generated_ids, result = _run_generation(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                tracker=tracker,
                use_cache=use_cache,
            )
            tokenizer.decode(generated_ids.squeeze(0).tolist())
            del generated_ids
            assert result is not None
            rows.append(result)
            logger.info(
                "Prompt %d run %d/%d: wall %.3fs, %.2f tok/s, MFU: %s",
                length,
                run_index + 1,
                repeats,
                result.generation_wall_time_sec,
                result.generation_wall_throughput_tok_per_sec,
                f"{result.mfu_percent:.2f}%" if result.mfu_percent is not None else "N/A",
            )
        runs_by_length[length] = rows
        del prompt_ids

    report = _render_report(
        model_name=model_path.name,
        num_params=num_params,
        device=target_device,
        dtype=dtype,
        peak_tflops=peak_tflops,
        prompt_lengths=prompt_lengths,
        runs_by_length=runs_by_length,
        use_cache=use_cache,
    )

    resolved_raw_dir = raw_dir if raw_dir.is_absolute() else _repo_root / raw_dir
    md_dir = resolved_raw_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    md_prefix = "kvcache" if use_cache else "baseline"
    md_filename = f"{md_prefix}_latency_{model_tag}.md" if not model_tag.startswith(md_prefix) else f"latency_{model_tag}.md"
    md_path = md_dir / md_filename
    md_path.write_text(report, encoding="utf-8")
    logger.info("Saved raw latency markdown to %s", md_path)

    json_path = None
    if save_json:
        json_dir = resolved_raw_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{md_prefix}_latency_{model_tag}.json" if not model_tag.startswith(md_prefix) else f"latency_{model_tag}.json"
        json_path = json_dir / json_filename
        json_payload = {
            "schema_version": 1,
            "metadata": metadata,
            "cases": [
                {
                    "prompt_tokens": length,
                    "prompt_ids": prompts_by_length[length],
                    "runs": [result.to_dict() for result in runs_by_length[length]],
                    "summary": aggregate_results(runs_by_length[length]),
                }
                for length in prompt_lengths
            ],
        }
        json_path.write_text(json.dumps(json_payload, indent=2, allow_nan=False), encoding="utf-8")
        logger.info("Saved raw latency JSON to %s", json_path)

    print("\n" + report)
    return md_path, json_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="StepInfer single-request latency benchmark")
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
        "--prompt-lengths",
        type=_positive_int,
        nargs="+",
        default=DEFAULT_PROMPT_LENGTHS,
        help="Exact synthetic input token lengths",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=32,
        help="Fixed number of generated tokens per run; EOS stopping is disabled",
    )
    parser.add_argument(
        "--warmup-runs",
        type=_positive_int,
        default=3,
        help="Full-generation warmup runs for each prompt-length case",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=10,
        help="Measured repetitions for each prompt-length case",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic token IDs")
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
        help="Disable KV Cache to run baseline full recalculation",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON raw artifacts, keep only Markdown",
    )

    args = parser.parse_args()
    run_latency_benchmark(
        model_path=args.model_path,
        device=args.device,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        seed=args.seed,
        tag=args.tag,
        raw_dir=args.raw_dir,
        use_cache=not args.no_cache,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
