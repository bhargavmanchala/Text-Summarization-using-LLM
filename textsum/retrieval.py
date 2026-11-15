"""A dependency-free BM25 retriever for passage search."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were "
    "what when where which who why with how do does did".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Hit:
    index: int
    score: float
    text: str


class BM25Index:
    """Okapi BM25 over a fixed list of passages."""

    def __init__(self, passages: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(p)) for p in passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if passages else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(passages)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
                score += self._idf[term] * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append(Hit(i, score, self.passages[i]))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]
