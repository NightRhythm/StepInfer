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

    def encode(self, text: str) -> Tensor:
        """将单条输入文本编码为模型输入的 token IDs。

        异常：
            TypeError: text 类型不正确，或 tokenizer 返回的结果不是 Tensor。
            ValueError: 编码结果的形状不符合单条模型输入要求。

        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

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

    def decode(self, token_ids: Tensor) -> str:
        """将模型生成的 token IDs 解码为文本。

        异常：
            TypeError: token_ids 类型不正确，或 tokenizer 返回的结果不是字符串。
            ValueError: token_ids 的形状不符合单条解码要求。
        """

        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if (
            token_ids.ndim != 2
            or token_ids.shape[0] != 1
            or token_ids.shape[1] == 0
        ):
            raise ValueError(
                "token_ids must have a non-empty shape [1, sequence_length]"
            )

        text = self.tokenizer.decode(token_ids=token_ids)
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode() must return a string for single-sequence input")

        return text

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 50,
    ) -> dict[str, object]:
        """根据单条输入文本生成文本。

        异常：
            TypeError: prompt 不是字符串。
            ValueError: prompt 为空、只包含空白字符，或 max_new_tokens 不是正数。
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty or whitespace-only")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")

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

