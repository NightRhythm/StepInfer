"""单请求生成基准测试，并提供独立的 profiling 模式。"""

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

from benchmarks.metrics import BenchmarkResult, BenchmarkTracker, aggregate_results
from stepinfer.config.model_config import ModelConfig
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.sampler import sample_next_token
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer
from stepinfer.utils.profiler import TorchProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_LENGTHS = [32, 64, 256, 512, 1024]


def _run_generation(
    model: Qwen3ForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    tracker: BenchmarkTracker | None = None,
) -> tuple[torch.Tensor, BenchmarkResult | None]:
    """从精确的 token ID 执行固定长度贪心解码。

    分词、主机到设备的数据传输和文本解码不计入本模型生成基准。
    关闭 EOS 停止，使每次实验执行相同数量的生成步骤。
    """
    input_ids = prompt_ids
    if tracker is not None:
        tracker.start()

    with torch.inference_mode():
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


def _range_text(values: list[float]) -> str:
    return f"{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def _render_report(
    *,
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    prompt_lengths: list[int],
    max_new_tokens: int,
    warmup_runs: int,
    repeats: int,
    seed: int,
    runs_by_length: dict[int, list[BenchmarkResult]],
) -> str:
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    lines = [
        "# 生成性能基准报告",
        "",
        "## 测量口径",
        "",
        "- 输入采用固定随机种子生成的合成 token ID 张量，报告中的输入长度为实际 token 数。",
        f"- 每轮固定生成 {max_new_tokens} 个 token，关闭 EOS 提前停止。",
        f"- 每个输入长度先执行 {warmup_runs} 轮完整生成预热，再执行 {repeats} 轮正式测量。",
        "- 计时不包含分词、输入从主机传到设备以及文本解码。",
        "- 首 token 和 ITL 在 CUDA 上使用 Event 时间间隔，在 CPU 上使用墙钟步骤间隔；这些指标不是端到端 TTFT。",
        "- CUDA Event 间隔可能包含等待 CPU 提交工作的设备空闲时间，不等于 kernel 耗时之和。",
        "- 生成墙钟耗时包含 Python 生成循环和最后的设备同步。",
        "- 逐步骤埋点有额外开销，这里的墙钟计时反映带埋点的生成循环。",
        "- ITL 分位数在每轮请求内部使用线性插值；汇总表报告各轮分位数的中位数，不是合并请求后的尾延迟。",
        "- allocated 峰值包含模型权重；峰值增量减去运行前的 allocated 基线。",
        "- 缓存分配器可能跨测试用例保留 reserved 显存，应结合起始基线和运行顺序解读。",
        "- Profiler 独立运行，不计入本报告的性能结果。",
        "",
        "## 运行环境",
        "",
        f"- Model path: `{model_path}`",
        f"- Device: `{device}` ({device_name})",
        f"- Dtype: `{dtype}`",
        f"- Python: `{platform.python_version()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- CUDA runtime: `{torch.version.cuda or 'N/A'}`",
        f"- Prompt lengths: `{prompt_lengths}`",
        f"- Seed: `{seed}`",
        "",
        "## 汇总结果",
        "",
        "数值为正式测量的中位数，方括号内为观测到的最小值和最大值。",
        "",
        "| Prompt tokens | Output tokens | Generation wall (s) | First-token interval (ms) | Avg ITL (ms) | P50 ITL (ms) | P90 ITL (ms) | P99 ITL (ms) | Wall throughput (tok/s) | Peak allocated (MiB) | Peak delta (MiB) | Peak reserved (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for length in prompt_lengths:
        rows = runs_by_length[length]
        def range_for(name: str) -> str:
            values = [getattr(row, name) for row in rows]
            present = [float(value) for value in values if value is not None]
            return _range_text(present) if present else "N/A"

        p50 = range_for("p50_itl_ms")
        p90 = range_for("p90_itl_ms")
        p99 = range_for("p99_itl_ms")

        lines.append(
            "| "
            + " | ".join(
                [
                    str(length),
                    str(rows[0].generated_tokens),
                    range_for("generation_wall_time_sec"),
                    range_for("first_token_interval_ms"),
                    range_for("avg_itl_ms"),
                    p50,
                    p90,
                    p99,
                    range_for("generation_wall_throughput_tok_per_sec"),
                    range_for("peak_allocated_mb"),
                    range_for("peak_allocated_delta_mb"),
                    range_for("peak_reserved_mb"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 逐轮结果",
            "",
            "| Prompt tokens | Run | Generated tokens | Wall time (s) | First-token interval (ms) | Avg ITL (ms) | Wall throughput (tok/s) | Allocated before (MiB) | Peak allocated (MiB) | Peak allocated delta (MiB) | Reserved before (MiB) | Peak reserved (MiB) | Free after (MiB) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for length in prompt_lengths:
        for run_index, row in enumerate(runs_by_length[length], start=1):
            lines.append(
                f"| {length} | {run_index} | {row.generated_tokens} | "
                f"{row.generation_wall_time_sec:.6f} | {row.first_token_interval_ms:.3f} | "
                f"{row.avg_itl_ms if row.avg_itl_ms is not None else 'N/A'} | "
                f"{row.generation_wall_throughput_tok_per_sec:.3f} | {row.allocated_before_mb:.2f} | "
                f"{row.peak_allocated_mb:.2f} | {row.peak_allocated_delta_mb:.2f} | "
                f"{row.reserved_before_mb:.2f} | {row.peak_reserved_mb:.2f} | "
                f"{row.device_free_after_mb:.2f} |"
            )

    return "\n".join(lines) + "\n"


def run_benchmark(
    model_path: Path,
    device: str,
    prompt_lengths: list[int],
    max_new_tokens: int = 32,
    warmup_runs: int = 3,
    repeats: int = 10,
    seed: int = 42,
    enable_profiler: bool = False,
    trace_dir: Path | None = None,
    report_path: Path = Path("docs/benchmarks/generation-benchmark.md"),
) -> None:
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

    logger.info(
        "Model ready: %d parameters, device=%s, dtype=%s",
        sum(parameter.numel() for parameter in model.parameters()),
        device,
        dtype,
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
        "model_path": str(model_path),
        "device": str(target_device),
        "dtype": str(dtype),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "seed": seed,
        "prompt_lengths": prompt_lengths,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "mode": "profile" if enable_profiler else "benchmark",
        "batch_size": 1,
        "ignore_eos": True,
        "scope": "resident input IDs -> fixed-length generation; excludes text encode/decode",
        "timing_backend": "cuda_event" if use_cuda else "cpu_perf_counter",
        "quantile_method": "linear interpolation at (n - 1) * q",
        "allocator_policy": "no empty_cache between cases; peaks reset after synchronization",
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
        logger.info(
            "Prompt length %d: %d warmup runs, %d measured runs%s",
            length,
            warmup_runs,
            repeats,
            " (profile mode)" if enable_profiler else "",
        )

        for warmup_index in range(warmup_runs):
            logger.info("Warmup %d/%d for prompt length %d", warmup_index + 1, warmup_runs, length)
            warmup_ids, _ = _run_generation(model, prompt_ids, max_new_tokens=max_new_tokens)
            tokenizer.decode(warmup_ids.squeeze(0).tolist())
            del warmup_ids

        if enable_profiler:
            profiler = TorchProfiler(use_cuda=use_cuda, record_shapes=True, profile_memory=True)
            with profiler:
                profiled_ids, _ = _run_generation(model, prompt_ids, max_new_tokens=max_new_tokens)
            tokenizer.decode(profiled_ids.squeeze(0).tolist())
            del profiled_ids
            logger.info("Profiler operations for prompt length %d:\n%s", length, profiler.summary(row_limit=15))
            if trace_dir is not None:
                resolved_trace_dir = trace_dir if trace_dir.is_absolute() else _repo_root / trace_dir
                trace_path = resolved_trace_dir / f"trace_prompt_{length}.json"
                profiler.export_chrome_trace(trace_path)
                trace_path.with_suffix(".metadata.json").write_text(
                    json.dumps({"metadata": metadata, "prompt_ids": prompts_by_length[length]}, indent=2),
                    encoding="utf-8",
                )
                logger.info("Saved trace to %s", trace_path)
            del prompt_ids, profiler
            continue

        rows: list[BenchmarkResult] = []
        for run_index in range(repeats):
            tracker = BenchmarkTracker(device=target_device)
            generated_ids, result = _run_generation(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                tracker=tracker,
            )
            tokenizer.decode(generated_ids.squeeze(0).tolist())
            del generated_ids
            assert result is not None
            rows.append(result)
            logger.info(
                "Prompt %d run %d/%d: generation wall %.3fs, %.2f tok/s",
                length,
                run_index + 1,
                repeats,
                result.generation_wall_time_sec,
                result.generation_wall_throughput_tok_per_sec,
            )
        runs_by_length[length] = rows
        del prompt_ids

    if enable_profiler:
        logger.info("Profiling finished. No benchmark report was written because profiling adds measurement overhead.")
        return

    report = _render_report(
        model_path=model_path,
        device=target_device,
        dtype=dtype,
        prompt_lengths=prompt_lengths,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        repeats=repeats,
        seed=seed,
        runs_by_length=runs_by_length,
    )
    resolved_report_path = report_path if report_path.is_absolute() else _repo_root / report_path
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    report += "\n## 复现实验元数据\n\n```json\n" + json.dumps(metadata, indent=2) + "\n```\n"
    resolved_report_path.write_text(report, encoding="utf-8")
    raw_path = resolved_report_path.with_suffix(".json")
    raw_path.write_text(
        json.dumps({
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
        }, indent=2, allow_nan=False), encoding="utf-8",
    )
    logger.info("Saved benchmark report to %s", resolved_report_path)
    print(report)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="StepInfer single-request generation benchmark")
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiler-only cases; profiling data is not used as benchmark timing",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("traces"),
        help="Directory for Chrome Trace JSON files in profile mode",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/benchmarks/generation-benchmark.md"),
        help="Markdown report path for benchmark mode",
    )

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model_path,
        device=args.device,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        repeats=args.repeats,
        seed=args.seed,
        enable_profiler=args.profile,
        trace_dir=args.trace_dir if args.profile else None,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
