"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL = "MBZUAI/LaMini-Flan-T5-248M"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Attributes:
        model_name: Hugging Face model id or local path for the local model.
        google_api_key: Optional Gemini key; enables the Gemini Q&A backend.
        gemini_model: Gemini model name used when a key is set.
        tesseract_cmd: Optional path to the Tesseract executable (for OCR).
    """

    model_name: str = DEFAULT_MODEL
    google_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    tesseract_cmd: str | None = None

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.google_api_key)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from ``env`` (defaults to ``os.environ``)."""
    env = os.environ if env is None else env

    def get(name: str) -> str | None:
        value = env.get(name, "").strip()
        return value or None

    return Settings(
        model_name=get("SUMMARIZER_MODEL") or DEFAULT_MODEL,
        google_api_key=get("GOOGLE_API_KEY"),
        gemini_model=get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        tesseract_cmd=get("TESSERACT_CMD"),
    )
