"""Helpers for splitting text and token sequences into overlapping chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def chunk_sequence(items: Sequence[T], size: int, overlap: int = 0) -> list[Sequence[T]]:
    """Split ``items`` into windows of ``size`` that overlap by ``overlap``."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if len(items) == 0:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(items), step):
        chunks.append(items[start : start + size])
        if start + size >= len(items):
            break
    return chunks


def chunk_words(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split ``text`` into word-based chunks of roughly ``size`` words."""
    words = text.split()
    return [" ".join(c) for c in chunk_sequence(words, size, overlap)]


def word_count(text: str) -> int:
    return len(text.split())
