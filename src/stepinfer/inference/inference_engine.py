"""协调 tokenizer、模型和生成循环的基础推理入口。"""

import torch
from torch import Tensor, nn

from stepinfer.inference.generator import generate_tokens
from stepinfer.tokenizer.base import BaseTokenizer


class InferenceEngine:
    """提供从输入文本到生成文本的完整调用入口。"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        *,
        device: str | torch.device = "cpu",
        eos_token_id: int | None = None,
        sampling_method: str = "greedy",
    ) -> None:
        """保存推理所需组件，并完成基础校验。

        异常：
            TypeError: model 不是 ``torch.nn.Module``，或 tokenizer 缺少
                可调用的 encode() / decode() 方法。
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        if tokenizer is None:
            raise TypeError("tokenizer cannot be None")
        
        encode_fn = getattr(tokenizer, "encode", None)
        decode_fn = getattr(tokenizer, "decode", None)
        
        if not callable(encode_fn):
            raise TypeError("Tokenizer must provide a callable 'encode' method.")
        
        if not callable(decode_fn):
            raise TypeError("Tokenizer must provide a callable 'decode' method.")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(self.tokenizer, "eos_token_id", None)
        )
        self.sampling_method = sampling_method

        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str | list[str]) -> Tensor | tuple[Tensor, Tensor]:
        """将单条或批量输入文本编码为模型输入的 token IDs (单条) 或 (token IDs, padding mask) (批量)。

        异常：
            TypeError: text 类型不正确，或 tokenizer 返回的结果类型不匹配。
            ValueError: 文本为空、包含空白字符，或编码结果形状不合法。
        """
        if not isinstance(text, (str, list)):
            raise TypeError("text must be a string or a list of strings")

        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text cannot be empty or whitespace-only")

            input_ids = self.tokenizer.encode(text=text, return_tensors="pt")
            if not isinstance(input_ids, Tensor):
                raise TypeError("tokenizer.encode() must return a torch.Tensor")
            if (
                input_ids.ndim != 2
                or input_ids.shape[0] != 1
                or input_ids.shape[1] == 0
            ):
                raise ValueError(
                    "tokenizer.encode() must return a non-empty tensor with shape "
                    "[1, sequence_length]"
                )

            return input_ids

        # 批量文本处理
        if not text:
            raise ValueError("texts cannot be empty")
        for t in text:
            if not isinstance(t, str):
                raise TypeError("All items in texts must be strings")
            if not t.strip():
                raise ValueError("prompts cannot be empty or whitespace-only")

        result = self.tokenizer.encode(text=text, return_tensors="pt", return_attention_mask=True)
        if isinstance(result, tuple) and len(result) == 2:
            input_ids, attention_mask = result
            if isinstance(input_ids, Tensor) and isinstance(attention_mask, Tensor):
                return input_ids, attention_mask

        # 回退适配：若非标准 tokenizer 未返回 mask，则逐条 encode 并对齐左填充
        encoded_list = [self.tokenizer.encode(t, return_tensors="pt") for t in text]
        max_len = max(t.size(-1) for t in encoded_list)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", 0)
        batch_ids = []
        batch_masks = []
        for t in encoded_list:
            pad_len = max_len - t.size(-1)
            pads = torch.full((1, pad_len), pad_id, dtype=torch.long)
            batch_ids.append(torch.cat([pads, t], dim=-1))
            mask = torch.cat(
                [torch.zeros((1, pad_len), dtype=torch.long), torch.ones_like(t, dtype=torch.long)],
                dim=-1,
            )
            batch_masks.append(mask)
        return torch.cat(batch_ids, dim=0), torch.cat(batch_masks, dim=0)

    def decode(self, token_ids: Tensor) -> str | list[str]:
        """将模型生成的 token IDs 解码为单条文本或文本列表。

        异常：
            TypeError: token_ids 类型不正确。
            ValueError: token_ids 的形状不合法。
        """
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 2 or token_ids.shape[1] == 0 or token_ids.shape[0] == 0:
            raise ValueError(
                "token_ids must have a non-empty shape [batch_size, sequence_length]"
            )

        return self.tokenizer.decode(token_ids=token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str | list[str],
        *,
        max_new_tokens: int = 50,
    ) -> dict[str, object] | list[dict[str, object]]:
        """根据单条或批量输入文本自回归生成文本。

        异常：
            TypeError: prompt 不是字符串或字符串列表。
            ValueError: prompt 为空、只包含空白字符，或 max_new_tokens 不是正数。
        """
        if not isinstance(prompt, (str, list)):
            raise TypeError("prompt must be a string or a list of strings")

        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt cannot be empty or whitespace-only")

            input_ids = self.encode(prompt)
            predicted_token_ids, token_count, stop_reason = generate_tokens(
                model=self.model,
                input_ids=input_ids,
                device=self.device,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id,
                sampling_method=self.sampling_method,
            )
            decoded_text = self.decode(predicted_token_ids)
            return {
                "prompt": prompt,
                "generated_text": decoded_text,
                "sampling_method": self.sampling_method,
                "token_count": token_count,
                "stop_reason": stop_reason,
            }

        # -----------------------------------------------------------------
        # 批量请求处理分支 (Batch Generation with Left-Padding)
        # -----------------------------------------------------------------
        input_ids, padding_mask = self.encode(prompt)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        predicted_token_ids, token_counts, stop_reasons = generate_tokens(
            model=self.model,
            input_ids=input_ids,
            device=self.device,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=pad_id,
            padding_mask=padding_mask,
            sampling_method=self.sampling_method,
        )

        decoded_texts = self.decode(predicted_token_ids)
        if isinstance(decoded_texts, str):
            decoded_texts = [decoded_texts]

        results: list[dict[str, object]] = []
        for i, p in enumerate(prompt):
            results.append(
                {
                    "prompt": p,
                    "generated_text": decoded_texts[i],
                    "sampling_method": self.sampling_method,
                    "token_count": token_counts[i] if isinstance(token_counts, list) else token_counts,
                    "stop_reason": stop_reasons[i] if isinstance(stop_reasons, list) else stop_reasons,
                }
            )

        return results

