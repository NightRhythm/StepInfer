"""批处理生成延迟与吞吐量基准测试。

专注于测量不同 Batch Size (1, 2, 4, 8) 下的端到端生成延迟、吞吐量、ITL 及显存开销。
实测验证 GEMV 到 GEMM 的计算密度跃迁与 MFU 提升。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import sys
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

DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
DEFAULT_PROMPT_LENGTHS = [64, 256]


def _run_batch_generation(
    model: Qwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    tracker: BenchmarkTracker | None = None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, BenchmarkResult | None]:
    """从精确的 token ID 执行固定长度贪心解码（支持 batch_size >= 1）。

    分词、主机到设备的数据传输和文本解码不计入本模型生成基准。
    关闭 EOS 停止，使每次实验执行相同数量的生成步骤。
    """
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
                curr_pos = torch.full((batch_size, 1), kv_cache[0].current_length, device=device)
                logits = model(next_token.unsqueeze(-1), positions=curr_pos, kv_cache=kv_cache)
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
    return input_ids, tracker.finish(prompt_tokens=prompt_len, batch_size=batch_size)


def _make_batch_prompt_ids(
    batch_size: int,
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
        size=(batch_size, length),
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


def _render_batch_report(
    *,
    model_name: str,
    num_params: int,
    device: torch.device,
    dtype: torch.dtype,
    peak_tflops: float | None,
    batch_sizes: list[int],
    prompt_lengths: list[int],
    runs_by_case: dict[tuple[int, int], list[BenchmarkResult]],
    use_cache: bool = True,
) -> str:
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    dtype_str = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")
    peak_tflops_str = f"{peak_tflops:.1f}" if peak_tflops is not None else "N/A"
    cache_str = "Static pre-allocated (in-place slicing)" if use_cache else "None"

    lines = [
        f"# Batch Latency & Throughput Benchmark — {model_name}",
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
        f"| Batch Sizes | {', '.join(str(b) for b in batch_sizes)} |",
        "| Sampling | Greedy (argmax) |",
        "",
        "---",
        "",
        "## Results",
        "",
        "数值格式为 `中位数 [最小值, 最大值]`。",
        "",
        "| Batch Size | Prompt Len | Output Tok/seq | Total Tok | Latency (s) | Throughput (tok/s) | Speedup vs B1 | Avg ITL (ms) | P50 ITL (ms) | P90 ITL (ms) | Peak Mem (MB) | Delta Mem (MB) | MFU (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for p_len in prompt_lengths:
        b1_median_throughput = None
        if (1, p_len) in runs_by_case:
            b1_runs = runs_by_case[(1, p_len)]
            b1_median_throughput = statistics.median(r.generation_wall_throughput_tok_per_sec for r in b1_runs)

        for b_size in batch_sizes:
            key = (b_size, p_len)
            if key not in runs_by_case:
                continue
            rows = runs_by_case[key]

            def range_for(name: str, precision: int = 3) -> str:
                values = [getattr(row, name) for row in rows]
                present = [float(value) for value in values if value is not None]
                return _range_text(present, precision=precision) if present else "N/A"

            output_tokens_per_seq = rows[0].generated_tokens // b_size
            total_tokens = str(rows[0].generated_tokens)
            latency_str = range_for("generation_wall_time_sec", 3)
            throughput_str = range_for("generation_wall_throughput_tok_per_sec", 2)

            med_tp = statistics.median(r.generation_wall_throughput_tok_per_sec for r in rows)
            if b1_median_throughput and b1_median_throughput > 0:
                speedup_str = f"{med_tp / b1_median_throughput:.2f}x"
            else:
                speedup_str = "1.00x"

            avg_itl_str = range_for("avg_itl_ms", 2)
            p50_str = range_for("p50_itl_ms", 2)
            p90_str = range_for("p90_itl_ms", 2)
            peak_mem_str = range_for("peak_allocated_mb", 2)
            delta_mem_str = range_for("peak_allocated_delta_mb", 2)
            mfu_str = range_for("mfu_percent", 2)

            lines.append(
                f"| {b_size} | {p_len} | {output_tokens_per_seq} | {total_tokens} | {latency_str} | "
                f"{throughput_str} | {speedup_str} | {avg_itl_str} | {p50_str} | {p90_str} | "
                f"{peak_mem_str} | {delta_mem_str} | {mfu_str} |"
            )

    return "\n".join(lines) + "\n"


def _derive_model_tag(model_path: Path) -> str:
    name = model_path.name.lower()
    if "0.6b" in name:
        return "0.6b"
    if "1.7b" in name:
        return "1.7b"
    return name.replace("-", "_")


def run_batch_benchmark(
    model_path: Path,
    device: str,
    batch_sizes: list[int] | None = None,
    prompt_lengths: list[int] | None = None,
    max_new_tokens: int = 32,
    warmup_runs: int = 2,
    repeats: int = 5,
    seed: int = 42,
    tag: str | None = None,
    raw_dir: Path = Path("docs/benchmarks/raw"),
    use_cache: bool = True,
    save_json: bool = True,
) -> tuple[Path, Path | None]:
    if batch_sizes is None:
        batch_sizes = DEFAULT_BATCH_SIZES
    if prompt_lengths is None:
        prompt_lengths = DEFAULT_PROMPT_LENGTHS

    if not batch_sizes or any(b <= 0 for b in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not prompt_lengths or any(length <= 0 for length in prompt_lengths):
        raise ValueError("prompt_lengths must contain positive integers")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_runs <= 0:
        raise ValueError("warmup_runs must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

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
    runs_by_case: dict[tuple[int, int], list[BenchmarkResult]] = {}

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
        "batch_sizes": batch_sizes,
        "prompt_lengths": prompt_lengths,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "kv_cache": "static_preallocated" if use_cache else "none",
        "ignore_eos": True,
        "scope": "resident input IDs -> fixed-length batched generation; excludes text encode/decode",
        "timing_backend": "cuda_event" if use_cuda else "cpu_perf_counter",
    }
    if use_cuda:
        props = torch.cuda.get_device_properties(target_device)
        metadata.update(gpu_name=props.name, gpu_total_bytes=props.total_memory)

    total_cases = len(prompt_lengths) * len(batch_sizes)
    current_case_idx = 0

    for p_len in prompt_lengths:
        for b_size in batch_sizes:
            current_case_idx += 1
            case_key = (b_size, p_len)
            logger.info(
                "[%d/%d] Testing Batch Size %d, Prompt Length %d: %d warmup runs, %d measured runs",
                current_case_idx,
                total_cases,
                b_size,
                p_len,
                warmup_runs,
                repeats,
            )

            prompt_ids = _make_batch_prompt_ids(
                batch_size=b_size,
                length=p_len,
                vocab_size=config.vocab_size,
                generator=torch.Generator(device="cpu").manual_seed(seed + p_len * 100 + b_size),
                device=target_device,
            )

            for warmup_index in range(warmup_runs):
                warmup_ids, _ = _run_batch_generation(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                )
                tokenizer.decode(warmup_ids[0].tolist())
                del warmup_ids

            rows: list[BenchmarkResult] = []
            for run_index in range(repeats):
                tracker = BenchmarkTracker(
                    device=target_device,
                    num_params=num_params,
                    dtype=dtype,
                    peak_tflops=peak_tflops,
                )
                generated_ids, result = _run_batch_generation(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    tracker=tracker,
                    use_cache=use_cache,
                )
                tokenizer.decode(generated_ids[0].tolist())
                del generated_ids
                assert result is not None
                rows.append(result)
                logger.info(
                    "Batch %d (p_len=%d) run %d/%d: wall %.3fs, %.2f tok/s, ITL: %.2f ms, MFU: %s",
                    b_size,
                    p_len,
                    run_index + 1,
                    repeats,
                    result.generation_wall_time_sec,
                    result.generation_wall_throughput_tok_per_sec,
                    result.avg_itl_ms if result.avg_itl_ms is not None else 0.0,
                    f"{result.mfu_percent:.2f}%" if result.mfu_percent is not None else "N/A",
                )
            runs_by_case[case_key] = rows
            del prompt_ids

    report = _render_batch_report(
        model_name=model_path.name,
        num_params=num_params,
        device=target_device,
        dtype=dtype,
        peak_tflops=peak_tflops,
        batch_sizes=batch_sizes,
        prompt_lengths=prompt_lengths,
        runs_by_case=runs_by_case,
        use_cache=use_cache,
    )

    resolved_raw_dir = raw_dir if raw_dir.is_absolute() else _repo_root / raw_dir
    md_dir = resolved_raw_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    md_filename = f"batch_latency_{model_tag}.md"
    md_path = md_dir / md_filename
    md_path.write_text(report, encoding="utf-8")
    logger.info("Saved raw batch latency markdown to %s", md_path)

    json_path = None
    if save_json:
        json_dir = resolved_raw_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"batch_latency_{model_tag}.json"
        json_path = json_dir / json_filename
        json_payload = {
            "schema_version": 1,
            "metadata": metadata,
            "cases": [
                {
                    "batch_size": b_size,
                    "prompt_tokens": p_len,
                    "runs": [result.to_dict() for result in runs_by_case[(b_size, p_len)]],
                    "summary": aggregate_results(runs_by_case[(b_size, p_len)]),
                }
                for p_len in prompt_lengths
                for b_size in batch_sizes
            ],
        }
        json_path.write_text(json.dumps(json_payload, indent=2, allow_nan=False), encoding="utf-8")
        logger.info("Saved raw batch latency JSON to %s", json_path)

    print("\n" + report)
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="StepInfer batched request latency and throughput benchmark")
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
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_LENGTHS,
        help="Prompt token lengths to benchmark",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Fixed number of generated tokens per sequence",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs prior to benchmarking each case",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured runs per case",
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
        help="Disable KV Cache",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving raw json file",
    )

    args = parser.parse_args()
    run_batch_benchmark(
        model_path=args.model_path,
        device=args.device,
        batch_sizes=args.batch_sizes,
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
