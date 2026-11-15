"""PDF text extraction."""

from __future__ import annotations

from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or contains no extractable text."""


def extract_text(file: BinaryIO | str) -> str:
    """Return the text of every page of a PDF (path or binary file object)."""
    try:
        reader = PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as exc:
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc
    text = "\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise PdfExtractionError(
            "No selectable text found. The PDF may be scanned; try the OCR option."
        )
    return text
