"""延迟与吞吐量基准评测脚本 (Latency and Throughput Benchmark)。

用于测试单请求、无 KV Cache 模式下的基线性能（Baseline），
覆盖不同 Prompt 长度（64, 256, 512）下的 TTFT、ITL、吞吐量与峰值显存。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# 保证直接运行脚本时能自动解析项目根路径与 src
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch

from benchmarks.metrics import BenchmarkTracker
from stepinfer.config.model_config import load_model_config
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.inference.sampler import sample_next_token
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer
from stepinfer.utils.profiler import TorchProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_with_tracking(
    engine: InferenceEngine,
    prompt: str,
    max_new_tokens: int,
    tracker: BenchmarkTracker,
) -> tuple[str, int]:
    """带有高精度逐步打点的自回归生成，用于精确提取 TTFT 与 ITL 序列。"""
    encoded = engine.tokenizer.encode(prompt, return_tensors="pt")
    input_ids = encoded.to(engine.device)
    prompt_len = input_ids.shape[-1]

    tracker.start()
    token_count = 0

    with torch.inference_mode():
        while token_count < max_new_tokens:
            logits = engine.model(input_ids)
            last_logits = logits[:, -1, :]
            next_token = sample_next_token(last_logits, method="greedy")
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # 记录当前 Token 生成完成的硬件事件
            tracker.record_step()
            token_count += 1

            if engine.eos_token_id is not None and next_token.item() == engine.eos_token_id:
                break

    output_text = engine.tokenizer.decode(input_ids.squeeze(0).tolist())
    return output_text, prompt_len


def run_benchmark(
    model_path: Path,
    device: str,
    prompt_lengths: list[int],
    max_new_tokens: int = 32,
    enable_profiler: bool = False,
    trace_dir: Path | None = None,
) -> None:
    logger.info("Initializing model and tokenizer from: %s", model_path)
    config = Qwen3Config.from_json_file(model_path / "config.json")
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if is_cuda else torch.float32

    model = Qwen3ForCausalLM(config=config)
    load_model_weights(model=model, model_path=model_path, device=device, dtype=dtype)
    tokenizer = HFTokenizer(load_model_config())

    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=device)
    num_params = sum(p.numel() for p in model.parameters())

    logger.info("Model loaded successfully (%d parameters, device: %s, dtype: %s)", num_params, device, dtype)

    # 1. Warmup
    logger.info("Running warmup forward step...")
    _ = engine.generate("Hello world", max_new_tokens=4)

    tracker = BenchmarkTracker(device=device, num_params=num_params)
    results = []

    # 2. 遍历各 Prompt 长度进行基准评测
    for length in prompt_lengths:
        logger.info("--- Benchmarking Prompt Length: %d tokens (max_new_tokens: %d) ---", length, max_new_tokens)

        # 构造指定长度的合成 Prompt
        raw_token_ids = torch.randint(100, config.vocab_size - 100, (length,)).tolist()
        synth_prompt = tokenizer.decode(raw_token_ids)

        if enable_profiler:
            profiler = TorchProfiler(use_cuda=is_cuda, record_shapes=True)
            with profiler:
                _, actual_prompt_len = generate_with_tracking(engine, synth_prompt, max_new_tokens, tracker)
            logger.info("Profiler top operations for prompt length %d:\n%s", length, profiler.summary(row_limit=10))
            if trace_dir:
                trace_path = trace_dir / f"trace_prompt_{length}.json"
                profiler.export_chrome_trace(trace_path)
                logger.info("Saved trace to: %s", trace_path)
        else:
            _, actual_prompt_len = generate_with_tracking(engine, synth_prompt, max_new_tokens, tracker)

        res = tracker.finish(prompt_tokens=actual_prompt_len)
        results.append(res)
        logger.info("\n%s\n", res.summary())

    # 3. 打印 Markdown 总结表格
    header = (
        f"| {'Prompt Len':^12} | {'Gen Toks':^10} | {'TTFT (ms)':^12} | "
        f"{'Avg ITL (ms)':^14} | {'P90 ITL (ms)':^14} | {'Throughput (t/s)':^18} | {'Peak VRAM (MB)':^16} |"
    )
    separator = f"|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*16}|{'-'*16}|{'-'*20}|{'-'*18}|"
    print("\n" + "=" * 115)
    print("Baseline Benchmark Summary (No KV Cache)")
    print("=" * 115)
    print(header)
    print(separator)
    for r in results:
        print(
            f"| {r.prompt_tokens:^12} | {r.generated_tokens:^10} | {r.ttft_ms:^12.2f} | "
            f"{r.avg_itl_ms:^14.2f} | {r.p90_itl_ms:^14.2f} | {r.throughput_tok_per_sec:^18.2f} | "
            f"{r.peak_memory_mb:^16.2f} |"
        )
    print("=" * 115 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="StepInfer Latency & Throughput Benchmark")
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
        help="Device to run benchmark on (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[64, 256, 512],
        help="List of prompt lengths to test",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of new tokens to generate per prompt",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler to record operators and shapes",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("traces"),
        help="Directory to save Chrome Trace JSON files",
    )

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model_path,
        device=args.device,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        enable_profiler=args.profile,
        trace_dir=args.trace_dir if args.profile else None,
    )


if __name__ == "__main__":
    main()
