"""Local seq2seq model wrapper (LaMini-Flan-T5 by default)."""

from __future__ import annotations


class LocalSeq2Seq:
    """Wrapper around a Hugging Face seq2seq model (loaded on construction)."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(self, prompt: str, *, max_input_tokens: int = 1024, **gen_kwargs) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
