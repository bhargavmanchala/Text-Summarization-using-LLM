import io

import pytest

from textsum.chunking import chunk_sequence, chunk_words
from textsum.config import DEFAULT_MODEL, load_settings
from textsum.pdf_io import PdfExtractionError, extract_text
from textsum.qa import NOT_FOUND, DocumentQA
from textsum.retrieval import BM25Index
from textsum.summarizer import Summarizer


class FakeModel:
    """Tokenizes on whitespace; 'summarizes' by keeping the first 3 words."""

    def __init__(self):
        self.prompts = []

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        body = prompt.split("\n\n", 1)[-1]
        return " ".join(body.split()[:3])


def test_chunk_sequence_overlap_and_coverage():
    chunks = chunk_sequence(list(range(10)), size=4, overlap=1)
    assert chunks == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]


def test_chunk_sequence_validation():
    with pytest.raises(ValueError):
        chunk_sequence([1], size=0)
    with pytest.raises(ValueError):
        chunk_sequence([1], size=2, overlap=2)
    assert chunk_sequence([], size=3) == []


def test_chunk_words():
    assert chunk_words("a b c d e", size=3, overlap=1) == ["a b c", "c d e"]


def test_settings_defaults_and_env():
    assert load_settings({}).model_name == DEFAULT_MODEL
    assert not load_settings({}).gemini_enabled
    s = load_settings({"GOOGLE_API_KEY": " k ", "SUMMARIZER_MODEL": "m"})
    assert s.gemini_enabled and s.google_api_key == "k" and s.model_name == "m"


def test_bm25_ranks_relevant_passage_first():
    index = BM25Index(["cats purr softly", "the moon orbits earth", "dogs bark loudly"])
    hits = index.search("what does the moon orbit", top_k=2)
    assert hits[0].index == 1
    assert index.search("zzz unknown") == []


def test_summarizer_short_text_single_pass():
    model = FakeModel()
    text = " ".join(f"w{i}" for i in range(50))
    assert Summarizer(model).summarize(text, "short") == "w0 w1 w2"
    assert len(model.prompts) == 1


def test_summarizer_map_reduce_for_long_text():
    model = FakeModel()
    text = " ".join(f"w{i}" for i in range(2000))
    out = Summarizer(model).summarize(text)
    assert len(model.prompts) > 1
    assert len(out.split()) < 200


def test_summarizer_rejects_bad_input():
    s = Summarizer(FakeModel())
    with pytest.raises(ValueError):
        s.summarize("too short")
    with pytest.raises(ValueError):
        s.summarize("word " * 50, length="huge")


def test_document_qa_not_found_and_local_answer():
    local = FakeModel()
    qa = DocumentQA("Paris is the capital of France. " * 5, load_settings({}), local)
    assert qa.ask("quantum chromodynamics").text == NOT_FOUND
    answer = qa.ask("capital of France")
    assert answer.backend == "local" and answer.sources


def test_extract_text_rejects_garbage_and_empty_pdf():
    with pytest.raises(PdfExtractionError):
        extract_text(io.BytesIO(b"not a pdf"))
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.write(buf)
    buf.seek(0)
    with pytest.raises(PdfExtractionError):
        extract_text(buf)


def test_extract_text_from_repo_pdf():
    from pathlib import Path

    pdf = Path(__file__).parent.parent / "docs/report/01-problem-statement.pdf"
    assert len(extract_text(str(pdf)).split()) > 20
