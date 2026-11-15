"""Map-reduce summarization that handles documents of any length."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .chunking import chunk_sequence

# Output length presets: (min_new_tokens, max_new_tokens) per chunk summary.
LENGTHS: dict[str, tuple[int, int]] = {
    "short": (20, 80),
    "medium": (40, 160),
    "long": (80, 300),
}
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 40
MAX_ROUNDS = 3
MIN_INPUT_WORDS = 20


class TextModel(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def generate(self, prompt: str, **kwargs) -> str: ...


class Summarizer:
    """Summarize long text by summarizing chunks, then the summaries."""

    def __init__(self, model: TextModel) -> None:
        self.model = model

    def summarize(
        self,
        text: str,
        length: str = "medium",
        progress: Callable[[float], None] | None = None,
    ) -> str:
        if length not in LENGTHS:
            raise ValueError(f"length must be one of {sorted(LENGTHS)}")
        if len(text.split()) < MIN_INPUT_WORDS:
            raise ValueError(f"Provide at least {MIN_INPUT_WORDS} words to summarize.")
        min_new, max_new = LENGTHS[length]

        ids = self.model.encode(text)
        joined = ""
        for round_no in range(MAX_ROUNDS):
            chunks = chunk_sequence(ids, CHUNK_TOKENS, CHUNK_OVERLAP)
            summaries = []
            for i, chunk in enumerate(chunks):
                chunk_text = self.model.decode(list(chunk))
                summaries.append(self._summarize_chunk(chunk_text, len(chunk), min_new, max_new))
                if progress:
                    progress((round_no + (i + 1) / len(chunks)) / MAX_ROUNDS)
            joined = " ".join(s for s in summaries if s)
            if len(chunks) == 1:
                break
            ids = self.model.encode(joined)
        if progress:
            progress(1.0)
        return joined

    def _summarize_chunk(self, chunk_text: str, n_tokens: int, min_new: int, max_new: int) -> str:
        return self.model.generate(
            f"Summarize the following text:\n\n{chunk_text}",
            max_new_tokens=max_new,
            min_new_tokens=min(min_new, max(5, n_tokens // 4)),
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            early_stopping=True,
        )
