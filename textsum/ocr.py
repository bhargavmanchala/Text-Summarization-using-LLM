"""Optical character recognition for images (requires Tesseract)."""

from __future__ import annotations

from typing import BinaryIO


class OcrUnavailableError(RuntimeError):
    """Raised when Tesseract or its Python bindings are missing."""


def image_to_text(image: BinaryIO | str, tesseract_cmd: str | None = None) -> str:
    """Extract text from an image file using Tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OcrUnavailableError("Install 'pytesseract' and 'Pillow' to use OCR.") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        with Image.open(image) as img:
            return pytesseract.image_to_string(img).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailableError(
            "Tesseract is not installed or not on PATH. Set TESSERACT_CMD to its location."
        ) from exc
