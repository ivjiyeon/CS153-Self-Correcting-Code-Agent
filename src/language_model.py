"""Thin wrapper around a local Hugging Face chat model (e.g. Qwen2.5-Coder)."""
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    """Output of one model call plus token accounting."""
    text: str
    prompt_tokens: int
    completion_tokens: int


class LocalCodeModel:
    """Loads a chat LLM once and exposes a single `generate` method."""

    def __init__(self, model_name: str, dtype_name: str = "bfloat16"):
        """Load tokenizer and weights.

        Input:  model_name (str) - Hugging Face id, dtype_name (str) - 'bfloat16' | 'float16' | 'float32'.
        Output: None.
        """
        dtype_lookup = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype_lookup[dtype_name], device_map="auto")
        self.model.eval()

    def generate(self, chat_messages: List[Dict[str, str]], max_new_tokens: int,
                 temperature: float) -> GenerationResult:
        """Generate one reply to a chat history.

        Input:  chat_messages (List[dict]) - [{"role": ..., "content": ...}, ...],
                max_new_tokens (int), temperature (float) - 0.0 means greedy decoding.
        Output: GenerationResult(text, prompt_tokens, completion_tokens).
        """
        prompt_text = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True)
        encoded_prompt = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = encoded_prompt["input_ids"].shape[1]

        generation_arguments = {"max_new_tokens": max_new_tokens,
                                "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            generation_arguments.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            generation_arguments.update(do_sample=False)

        with torch.no_grad():
            output_ids = self.model.generate(**encoded_prompt, **generation_arguments)
        completion_ids = output_ids[0, prompt_length:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(completion_text, prompt_length, int(completion_ids.shape[0]))
