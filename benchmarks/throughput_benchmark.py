"""连续批处理综合吞吐量与系统级性能基准测试 (Continuous Batching Throughput Benchmark)。

对比两类批处理架构在异构非齐次工作负载 (长短不一真实请求流) 下的性能表现:
1. 静态批处理 (Static Batching Baseline): 严格按固定 Batch 对齐填充，短请求陪跑直到长请求完成;
2. 连续批处理 (Continuous Batching): 迭代级槽位调度 (Iteration-level Scheduling)，短请求完成即刻释放插槽，排队请求无缝插队复用。

度量核心工业级指标:
- 总服务耗时 (Makespan / Total Time, s)
- 有效 Token 吞吐量 (Effective Throughput, tokens/sec)
- 端到端加速比 (Speedup)
- 单请求时延分布 (Average, P50, P90, Min, Max Latency, s)
- 逐请求原始明细 (Prompt Tokens, Output Tokens, Latency Delta)
- GPU 显存峰值 (Peak Memory, MB)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import torch
from torch import nn

from benchmarks.metrics import get_peak_tflops
from stepinfer.config.model_config import load_model_config
from stepinfer.config.qwen3_config import Qwen3Config
from stepinfer.inference.inference_engine import InferenceEngine
from stepinfer.models import Qwen3ForCausalLM, load_model_weights
from stepinfer.tokenizer.hf_tokenizer import HFTokenizer
from stepinfer.utils.gpu_monitor import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("continuous_benchmark")

# 真实非均匀请求模板库 (短/中/长不同任务类型混合)
WORKLOAD_TEMPLATES = [
    # 极短请求 (常识/问答) -> 预期 10~15 tokens
    ("What is 15 + 27?", 12),
    ("The capital city of France is", 10),
    ("Who wrote the play Hamlet?", 15),
    ("What is the boiling point of water in Celsius?", 15),
    # 中等请求 (解释/摘要) -> 预期 30~50 tokens
    ("Explain the principle of conservation of energy in simple terms:", 40),
    ("Write a short three-line poem about the autumn wind and falling leaves:", 35),
    ("Summarize why the sky appears blue during the day:", 45),
    ("List three major benefits of regular physical exercise:", 40),
    # 较长请求 (代码/推理) -> 预期 70~100 tokens
    ("Write a Python function to check whether a string is a palindrome with comments:", 80),
    ("Describe the difference between supervised learning and unsupervised learning:", 90),
    ("Explain how a CPU cache works and why it is organized in L1, L2, and L3 levels:", 95),
    ("Write a step-by-step recipe for making classic Italian tomato pasta:", 85),
]


def generate_workload(num_requests: int, seed: int = 42) -> list[tuple[str, int]]:
    """生成具有长短差异的非齐次工作负载请求集合。"""
    rng = random.Random(seed)
    workload: list[tuple[str, int]] = []
    for i in range(num_requests):
        template_idx = i % len(WORKLOAD_TEMPLATES)
        prompt, target_max_tokens = WORKLOAD_TEMPLATES[template_idx]
        jitter = rng.randint(-3, 5)
        tokens_limit = max(8, target_max_tokens + jitter)
        workload.append((prompt, tokens_limit))
    return workload


def run_static_batching_baseline(
    model: nn.Module,
    tokenizer: Any,
    workload: list[tuple[str, int]],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """模拟静态批处理 Baseline:

    按固定 batch_size 一批批切分执行。在每批中，所有请求必须等待该批次中最长请求结束才能一起退场。
    """
    logger.info(f"Running Static Batching Baseline (batch_size={batch_size}, total_requests={len(workload)})...")
    monitor = GPUMonitor(device)
    monitor.reset_peak_stats()

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_batch_size=batch_size,
    )

    start_time = time.perf_counter()
    per_request_records: list[dict[str, Any]] = []

    num_chunks = math.ceil(len(workload) / batch_size)
    for chunk_idx in range(num_chunks):
        chunk = workload[chunk_idx * batch_size : (chunk_idx + 1) * batch_size]
        prompts = [item[0] for item in chunk]
        chunk_max_tokens = max(item[1] for item in chunk)

        results = engine.generate(prompts, max_new_tokens=chunk_max_tokens)
        chunk_finish_time = time.perf_counter() - start_time

        if isinstance(results, dict):
            results = [results]

        for req_idx_in_chunk, (res, original_item) in enumerate(zip(results, chunk)):
            global_req_idx = chunk_idx * batch_size + req_idx_in_chunk
            encoded_prompt = tokenizer.encode(original_item[0], return_tensors=None)
            prompt_len = (
                len(encoded_prompt.flatten().tolist())
                if isinstance(encoded_prompt, torch.Tensor)
                else len(encoded_prompt)
            )
            actual_tokens = min(res.get("token_count", 0), original_item[1])
            per_request_records.append({
                "req_id": global_req_idx,
                "prompt": original_item[0],
                "prompt_len": prompt_len,
                "target_tokens": original_item[1],
                "generated_tokens": actual_tokens,
                "latency_sec": chunk_finish_time,
            })

    total_wall_time = time.perf_counter() - start_time
    peak_mem_mb = monitor.snapshot().max_allocated_mb
    request_latencies = [r["latency_sec"] for r in per_request_records]
    total_tokens_generated = sum(r["generated_tokens"] for r in per_request_records)
    total_prompt_tokens = sum(r["prompt_len"] for r in per_request_records)

    sorted_lats = sorted(request_latencies)
    n = len(sorted_lats)
    p50_idx = int(math.ceil(0.5 * n)) - 1
    p90_idx = int(math.ceil(0.9 * n)) - 1

    return {
        "mode": "Static Batching",
        "total_wall_time_sec": total_wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens_generated,
        "throughput_tok_per_sec": total_tokens_generated / max(1e-6, total_wall_time),
        "avg_latency_sec": sum(request_latencies) / len(request_latencies),
        "p50_latency_sec": sorted_lats[max(0, p50_idx)],
        "p90_latency_sec": sorted_lats[max(0, p90_idx)],
        "min_latency_sec": sorted_lats[0],
        "max_latency_sec": sorted_lats[-1],
        "peak_mem_mb": peak_mem_mb,
        "requests": per_request_records,
    }


def run_continuous_batching_benchmark(
    model: nn.Module,
    tokenizer: Any,
    workload: list[tuple[str, int]],
    max_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """运行连续批处理 (Continuous Batching):

    将全部请求投入调度队列。引擎主循环以迭代级推进，短请求遇到目标步数或 EOS 立即退场，
    空出的 Slot 立即由等待队列中的新请求填补。
    """
    logger.info(f"Running Continuous Batching (max_batch_size={max_batch_size}, total_requests={len(workload)})...")
    monitor = GPUMonitor(device)
    monitor.reset_peak_stats()

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_batch_size=max_batch_size,
    )

    start_time = time.perf_counter()
    submitted_requests = []

    # 1. 全部请求投入连续调度队列
    for prompt, max_tokens in workload:
        req = engine.add_request(prompt=prompt, max_new_tokens=max_tokens)
        submitted_requests.append(req)

    # 2. 连续批步进循环，记录各个请求真实退场的时间戳
    completion_times: dict[str, float] = {}
    while any(not r.is_finished for r in submitted_requests):
        just_finished = engine.step()
        curr_time = time.perf_counter() - start_time
        for r in just_finished:
            if r.request_id not in completion_times:
                completion_times[r.request_id] = curr_time

    total_wall_time = time.perf_counter() - start_time
    peak_mem_mb = monitor.snapshot().max_allocated_mb

    per_request_records: list[dict[str, Any]] = []
    for idx, r in enumerate(submitted_requests):
        finish_t = completion_times[r.request_id]
        per_request_records.append({
            "req_id": idx,
            "prompt": r.prompt,
            "prompt_len": len(r.prompt_token_ids),
            "target_tokens": r.max_new_tokens,
            "generated_tokens": r.num_generated_tokens,
            "latency_sec": finish_t,
        })

    request_latencies = [rec["latency_sec"] for rec in per_request_records]
    total_tokens_generated = sum(rec["generated_tokens"] for rec in per_request_records)
    total_prompt_tokens = sum(rec["prompt_len"] for rec in per_request_records)

    sorted_lats = sorted(request_latencies)
    n = len(sorted_lats)
    p50_idx = int(math.ceil(0.5 * n)) - 1
    p90_idx = int(math.ceil(0.9 * n)) - 1

    return {
        "mode": "Continuous Batching",
        "total_wall_time_sec": total_wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens_generated,
        "throughput_tok_per_sec": total_tokens_generated / max(1e-6, total_wall_time),
        "avg_latency_sec": sum(request_latencies) / len(request_latencies),
        "p50_latency_sec": sorted_lats[max(0, p50_idx)],
        "p90_latency_sec": sorted_lats[max(0, p90_idx)],
        "min_latency_sec": sorted_lats[0],
        "max_latency_sec": sorted_lats[-1],
        "peak_mem_mb": peak_mem_mb,
        "requests": per_request_records,
    }


def format_markdown_report(
    static_res: dict[str, Any],
    cont_res: dict[str, Any],
    model_name: str,
    device_name: str,
    num_params: int,
    precision: str,
    num_requests: int,
    max_batch_size: int,
) -> str:
    """生成符合项目 raw 规范的纯净基准评测数据表格 (无主观总结与冗余声明)。"""
    speedup = static_res["total_wall_time_sec"] / max(1e-6, cont_res["total_wall_time_sec"])
    throughput_boost = (
        (cont_res["throughput_tok_per_sec"] - static_res["throughput_tok_per_sec"])
        / static_res["throughput_tok_per_sec"]
        * 100
    )
    avg_latency_reduction = (
        (static_res["avg_latency_sec"] - cont_res["avg_latency_sec"])
        / static_res["avg_latency_sec"]
        * 100
    )
    p50_latency_reduction = (
        (static_res["p50_latency_sec"] - cont_res["p50_latency_sec"])
        / static_res["p50_latency_sec"]
        * 100
    )
    p90_latency_reduction = (
        (static_res["p90_latency_sec"] - cont_res["p90_latency_sec"])
        / static_res["p90_latency_sec"]
        * 100
    )

    lines = [
        f"# Throughput Benchmark — {model_name}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|:---|:---|",
        f"| Model | {model_name} |",
        f"| Parameters | {num_params:,} |",
        f"| Precision | {precision} |",
        f"| Device | {device_name} |",
        f"| PyTorch | {torch.__version__} |",
        f"| Python | {platform.python_version()} |",
        "| Scheduler | Continuous Batching vs Static Baseline |",
        f"| Concurrency (Slots) | {max_batch_size} |",
        f"| Total Requests | {num_requests} |",
        "| Sampling | Greedy (argmax) |",
        "",
        "---",
        "",
        "## Results",
        "",
        "### 1. Aggregate Summary (汇总统计对比)",
        "",
        "| Metric | Static Batching | Continuous Batching | Delta |",
        "|:---|---:|---:|---:|",
        f"| Total Wall Time (s) | {static_res['total_wall_time_sec']:.3f} | {cont_res['total_wall_time_sec']:.3f} | {speedup:.2f}x speedup |",
        f"| Throughput (tok/s) | {static_res['throughput_tok_per_sec']:.2f} | {cont_res['throughput_tok_per_sec']:.2f} | +{throughput_boost:.1f}% |",
        f"| Avg Latency (s) | {static_res['avg_latency_sec']:.3f} | {cont_res['avg_latency_sec']:.3f} | -{avg_latency_reduction:.1f}% |",
        f"| P50 Latency (s) | {static_res['p50_latency_sec']:.3f} | {cont_res['p50_latency_sec']:.3f} | -{p50_latency_reduction:.1f}% |",
        f"| P90 Latency (s) | {static_res['p90_latency_sec']:.3f} | {cont_res['p90_latency_sec']:.3f} | -{p90_latency_reduction:.1f}% |",
        f"| Min Latency (s) | {static_res['min_latency_sec']:.3f} | {cont_res['min_latency_sec']:.3f} | {static_res['min_latency_sec'] / max(1e-6, cont_res['min_latency_sec']):.2f}x |",
        f"| Max Latency (s) | {static_res['max_latency_sec']:.3f} | {cont_res['max_latency_sec']:.3f} | {static_res['max_latency_sec'] / max(1e-6, cont_res['max_latency_sec']):.2f}x |",
        f"| Total Prompt Tokens | {static_res['total_prompt_tokens']} | {cont_res['total_prompt_tokens']} | - |",
        f"| Total Output Tokens | {static_res['total_tokens']} | {cont_res['total_tokens']} | +{(cont_res['total_tokens'] - static_res['total_tokens']) / max(1, static_res['total_tokens']) * 100:.1f}% |",
        f"| Peak Memory (MB) | {static_res['peak_mem_mb']:.1f} | {cont_res['peak_mem_mb']:.1f} | {cont_res['peak_mem_mb'] - static_res['peak_mem_mb']:+.1f} MB |",
        "",
        "### 2. Per-Request Breakdown (逐请求原始测量明细)",
        "",
        "| Req ID | Prompt Len | Target Tok | Output Tok (Static) | Output Tok (Cont) | Static Latency (s) | Continuous Latency (s) | Speedup | Latency Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s_req, c_req in zip(static_res["requests"], cont_res["requests"]):
        req_id = s_req["req_id"]
        prompt_len = s_req["prompt_len"]
        target_tok = s_req["target_tokens"]
        s_out = s_req["generated_tokens"]
        c_out = c_req["generated_tokens"]
        s_lat = s_req["latency_sec"]
        c_lat = c_req["latency_sec"]
        req_speedup = s_lat / max(1e-6, c_lat)
        delta_pct = (s_lat - c_lat) / max(1e-6, s_lat) * 100
        lines.append(
            f"| {req_id} | {prompt_len} | {target_tok} | {s_out} | {c_out} | {s_lat:.3f} | {c_lat:.3f} | {req_speedup:.2f}x | -{delta_pct:.1f}% |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Batching vs Static Batching Benchmark.")
    parser.add_argument("--model-path", type=str, default="assets/models/Qwen3-0.6B-Base", help="Path to model weights directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Max batch slots (concurrency)")
    parser.add_argument("--num-requests", type=int, default=8, help="Number of requests in the workload")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default=None, help="Model tag (e.g. 0.6b, 1.7b)")
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to output markdown report (defaults to docs/benchmarks/raw/md/throughput_<tag>.md)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output json artifact (defaults to docs/benchmarks/raw/json/throughput_<tag>.json)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 1. 载入模型与 Tokenizer
    model_path = Path(args.model_path).resolve()
    model_tag = args.tag or (
        "0.6b" if "0.6b" in model_path.name.lower() else "1.7b" if "1.7b" in model_path.name.lower() else "custom"
    )
    config = Qwen3Config.from_json_file(model_path / "config.json")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info("Initializing Qwen3ForCausalLM model...")
    model = Qwen3ForCausalLM(config=config)
    load_model_weights(model=model, model_path=model_path, device=str(device), dtype=model_dtype)
    tokenizer = HFTokenizer(load_model_config())

    num_params = sum(p.numel() for p in model.parameters())
    precision = "bf16" if model_dtype == torch.bfloat16 else "fp32"

    # 2. 生成非均匀混合请求集
    workload = generate_workload(num_requests=args.num_requests, seed=args.seed)
    logger.info(f"Generated workload with {len(workload)} requests:")
    for idx, (p, max_tok) in enumerate(workload):
        logger.info(f"  [{idx}] max_new={max_tok:2d} | Prompt: {p[:40]}...")

    # 3. 运行静态批处理基准 (Static Baseline)
    static_res = run_static_batching_baseline(
        model=model,
        tokenizer=tokenizer,
        workload=workload,
        batch_size=args.max_batch_size,
        device=device,
    )
    logger.info(
        f"[Static Baseline] Finished in {static_res['total_wall_time_sec']:.3f}s | "
        f"Throughput: {static_res['throughput_tok_per_sec']:.2f} tok/s | "
        f"Avg Latency: {static_res['avg_latency_sec']:.3f}s"
    )

    # 4. 运行连续批处理基准 (Continuous Batching)
    cont_res = run_continuous_batching_benchmark(
        model=model,
        tokenizer=tokenizer,
        workload=workload,
        max_batch_size=args.max_batch_size,
        device=device,
    )
    logger.info(
        f"[Continuous Batching] Finished in {cont_res['total_wall_time_sec']:.3f}s | "
        f"Throughput: {cont_res['throughput_tok_per_sec']:.2f} tok/s | "
        f"Avg Latency: {cont_res['avg_latency_sec']:.3f}s"
    )

    # 5. 生成报告并输出
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    report_md = format_markdown_report(
        static_res=static_res,
        cont_res=cont_res,
        model_name="Qwen3-0.6B-Base",
        device_name=device_name,
        num_params=num_params,
        precision=precision,
        num_requests=args.num_requests,
        max_batch_size=args.max_batch_size,
    )

    output_path = Path(args.output_md) if args.output_md else Path(f"docs/benchmarks/raw/md/throughput_{model_tag}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    logger.info(f"Benchmark raw markdown saved to: {output_path}")

    # 6. 保存原始 JSON 工件
    json_path = (
        Path(args.output_json)
        if args.output_json
        else Path(f"docs/benchmarks/raw/json/throughput_{model_tag}.json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "metadata": {
            "model_name": "Qwen3-0.6B-Base",
            "num_params": num_params,
            "precision": precision,
            "device": device_name,
            "max_batch_size": args.max_batch_size,
            "num_requests": args.num_requests,
            "seed": args.seed,
        },
        "static": static_res,
        "continuous": cont_res,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Benchmark raw json saved to: {json_path}")

    # 打印终端总结
    print("\n" + "=" * 60)
    print(report_md)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
