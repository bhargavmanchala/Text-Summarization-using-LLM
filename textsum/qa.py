"""Question answering over retrieved passages (Gemini or local model)."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_words
from .config import Settings
from .retrieval import BM25Index, Hit

NOT_FOUND = "I could not find that in the document."
_PROMPT = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, reply exactly: {nf}\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    backend: str


class DocumentQA:
    """Index a document once, then answer questions about it."""

    def __init__(self, text: str, settings: Settings, local_model=None) -> None:
        self.settings = settings
        self.local_model = local_model
        self.index = BM25Index(chunk_words(text))

    @property
    def backend(self) -> str:
        return f"gemini:{self.settings.gemini_model}" if self.settings.gemini_enabled else "local"

    def ask(self, question: str, top_k: int = 4) -> Answer:
        hits = self.index.search(question, top_k=top_k)
        if not hits:
            return Answer(NOT_FOUND, [], self.backend)
        prompt = _PROMPT.format(
            nf=NOT_FOUND,
            context="\n\n".join(h.text for h in hits),
            question=question,
        )
        return Answer(self._generate(prompt), hits, self.backend)

    def _generate(self, prompt: str) -> str:
        if self.settings.gemini_enabled:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.google_api_key)
            model = genai.GenerativeModel(self.settings.gemini_model)
            return model.generate_content(prompt).text.strip()
        if self.local_model is None:
            raise RuntimeError("No Gemini key set and no local model supplied.")
        return self.local_model.generate(prompt, max_new_tokens=150, num_beams=3)
