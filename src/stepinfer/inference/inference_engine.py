"""推理引擎入口模块 (InferenceEngine)。

协调 Tokenizer、模型与调度中枢，支持单请求与连续批处理 (Continuous Batching) 流水线自回归生成。
"""

import uuid
import torch
from torch import Tensor, nn

from stepinfer.cache.kv_cache import KVCache
from stepinfer.inference.sampler import sample_next_token
from stepinfer.scheduler.request import Request
from stepinfer.scheduler.scheduler import Scheduler
from stepinfer.tokenizer.base import BaseTokenizer


class InferenceEngine:
    """提供从输入文本到生成文本的完整调用入口，内置连续批处理调度流水线。"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        *,
        device: str | torch.device = "cpu",
        eos_token_id: int | None = None,
        sampling_method: str = "greedy",
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ) -> None:
        """初始化推理引擎，完成组件挂载与调度器就绪。"""
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if tokenizer is None:
            raise TypeError("tokenizer cannot be None")

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        self.sampling_method = sampling_method
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # 调度中枢
        self.scheduler = Scheduler(max_batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)

        # 显存预分配: 若模型配置就绪，预分配全局槽位隔离 KV Cache
        self.kv_cache: KVCache | None = None
        if hasattr(self.model, "config"):
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is None and hasattr(self.model, "parameters"):
                try:
                    model_dtype = next(self.model.parameters()).dtype
                except StopIteration:
                    model_dtype = torch.float32
            else:
                model_dtype = torch.float32

            self.kv_cache = KVCache.from_config(
                config=self.model.config,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                dtype=model_dtype,
                device=self.device,
            )

    def encode(self, text: str | list[str]) -> Tensor:
        """将输入文本编码为模型输入的 token IDs。"""
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text cannot be empty or whitespace-only")
            token_ids = self.tokenizer.encode(text=text, return_tensors="pt")
            if not isinstance(token_ids, Tensor):
                token_ids = torch.tensor(token_ids, dtype=torch.long)
            if token_ids.ndim == 1:
                token_ids = token_ids.unsqueeze(0)
            return token_ids.to(self.device)

        if not text:
            raise ValueError("texts cannot be empty")
        for t in text:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("prompts cannot be empty or whitespace-only")

        token_ids, _ = self.tokenizer.encode(text=text, return_tensors="pt")
        return token_ids.to(self.device)

    def decode(self, token_ids: Tensor) -> str | list[str]:
        """将 token IDs 解码为文本。"""
        if not isinstance(token_ids, Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)

    def add_request(self, prompt: str, max_new_tokens: int = 50) -> Request:
        """向调度队列提交一条新请求。"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty or whitespace-only.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")

        # 编码 Prompt
        encoded = self.tokenizer.encode(prompt, return_tensors=None)
        if isinstance(encoded, Tensor):
            prompt_tokens = encoded.flatten().tolist()
        else:
            prompt_tokens = list(encoded)

        req_id = f"req-{uuid.uuid4().hex[:8]}"
        req = Request(
            request_id=req_id,
            prompt=prompt,
            prompt_token_ids=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )
        self.scheduler.add_request(req)
        return req

    def step(self) -> list[Request]:
        """执行单步连续批调度与前向计算。

        返回:
            list[Request]: 本步刚触发完成的请求列表。
        """
        sched_res = self.scheduler.schedule()
        if sched_res.is_empty:
            return []

        requests = sched_res.requests

        with torch.inference_mode():
            if sched_res.is_prefill:
                # -------------------------------------------------------------
                # Prefill 阶段: 各新请求首步独立前向，写入各自的物理 slot
                # -------------------------------------------------------------
                next_tokens_list: list[int] = []
                for req in requests:
                    in_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
                    positions = torch.arange(len(req.prompt_token_ids), dtype=torch.long, device=self.device)

                    if self.kv_cache is not None:
                        logits = self.model(
                            in_ids,
                            positions=positions,
                            kv_cache=self.kv_cache,
                            slot_ids=[req.slot_id],
                        )
                    else:
                        logits = self.model(in_ids)

                    next_token = sample_next_token(logits[:, -1, :], method=self.sampling_method).item()
                    next_tokens_list.append(int(next_token))

                finished = self.scheduler.post_step(
                    requests=requests,
                    next_tokens=next_tokens_list,
                    eos_token_id=self.eos_token_id,
                )
                if self.kv_cache is not None:
                    for req in finished:
                        if req.slot_id is not None:
                            self.kv_cache.reset_slot(req.slot_id)
                return finished

            else:
                # -------------------------------------------------------------
                # Decode 阶段: 并发批处理各活跃序列的单步 Token 解码 (BMM 加速)
                # -------------------------------------------------------------
                curr_tokens = [req.generated_token_ids[-1] for req in requests]
                in_ids = torch.tensor(curr_tokens, dtype=torch.long, device=self.device).unsqueeze(-1)
                positions = torch.tensor([[req.total_len - 1] for req in requests], dtype=torch.long, device=self.device)
                slot_ids = [req.slot_id for req in requests]

                if self.kv_cache is not None:
                    logits = self.model(
                        in_ids,
                        positions=positions,
                        kv_cache=self.kv_cache,
                        slot_ids=slot_ids,
                    )
                else:
                    logits = self.model(in_ids)

                next_tokens = sample_next_token(logits[:, -1, :], method=self.sampling_method)
                next_tokens_list = next_tokens.flatten().tolist()

                finished = self.scheduler.post_step(
                    requests=requests,
                    next_tokens=next_tokens_list,
                    eos_token_id=self.eos_token_id,
                )
                if self.kv_cache is not None:
                    for req in finished:
                        if req.slot_id is not None:
                            self.kv_cache.reset_slot(req.slot_id)
                return finished

    def generate(
        self,
        prompt: str | list[str],
        *,
        max_new_tokens: int = 50,
    ) -> dict[str, object] | list[dict[str, object]]:
        """根据单条或批量输入文本自回归生成文本。"""
        if not isinstance(prompt, (str, list)):
            raise TypeError("prompt must be a string or a list of strings")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        if isinstance(prompt, str):
            prompts = [prompt]
            single = True
        else:
            prompts = prompt
            single = False

        if not prompts:
            raise ValueError("prompt cannot be empty")

        # 1. 提交所有请求至调度等待队列
        submitted_requests: list[Request] = []
        for p in prompts:
            req = self.add_request(p, max_new_tokens=max_new_tokens)
            submitted_requests.append(req)

        # 2. 连续批步进循环，直至所有提交的请求均触发完成
        while any(not r.is_finished for r in submitted_requests):
            self.step()

        # 3. 组织结果返回
        results: list[dict[str, object]] = []
        for r in submitted_requests:
            full_tokens = r.prompt_token_ids + r.generated_token_ids
            decoded_text = self.decode(torch.tensor([full_tokens], dtype=torch.long))
            if isinstance(decoded_text, list):
                decoded_text = decoded_text[0]

            stop_reason = (
                "Generated EOS token."
                if r.stop_reason == "stop"
                else f"Reached max_new_tokens limit of {r.max_new_tokens}."
            )
            results.append(
                {
                    "prompt": r.prompt,
                    "generated_text": decoded_text,
                    "sampling_method": self.sampling_method,
                    "token_count": r.num_generated_tokens,
                    "stop_reason": stop_reason,
                }
            )

        return results[0] if single else results
