# Text Summarization using LLM

A Streamlit app that **summarizes text, PDFs and images** with a local language model and lets you
**ask questions about your PDFs**. It started as a college mini-project and is now a single,
tested codebase.

[![CI](https://github.com/bhargavmanchala/Text-Summarization-using-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/bhargavmanchala/Text-Summarization-using-LLM/actions/workflows/ci.yml)

## Features

- **Summarize** pasted text, one or more PDFs, or an image (via OCR), with a short / medium / long
  length control. Long documents are handled by map-reduce over token chunks, so nothing is
  silently truncated.
- **Ask your PDF**: passages are retrieved with BM25 and answered by Google Gemini (if you set an API
  key) or by the same local model (no key, no cloud).
- **Runs locally by default**: summarization uses
  [`MBZUAI/LaMini-Flan-T5-248M`](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M); PDF text never
  leaves your machine unless you enable Gemini.
- Download the summary as a `.txt` file.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/bhargavmanchala/Text-Summarization-using-LLM.git
cd Text-Summarization-using-LLM
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first summary downloads the model (about 1 GB) and caches it. It runs on CPU, or on a GPU if
PyTorch can see one.

### Optional setup

| Feature | What to do |
|---|---|
| Gemini answers for PDF Q&A | Set `GOOGLE_API_KEY` (see [`.env.example`](.env.example)). Never commit the key. |
| Image OCR | Install [Tesseract](https://github.com/tesseract-ocr/tesseract); set `TESSERACT_CMD` if it is not on `PATH`. |
| Different model | Set `SUMMARIZER_MODEL` to a Hugging Face id or a local folder (any seq2seq model). |

PowerShell example:

```powershell
$env:GOOGLE_API_KEY = "your-key"
streamlit run app.py
```

## Project layout

```
app.py                  Streamlit UI
textsum/                Core library (no UI code)
  summarizer.py         Map-reduce summarization
  qa.py, retrieval.py   PDF question answering (BM25 + Gemini/local)
  pdf_io.py, ocr.py     Text extraction
  models.py, config.py  Model wrapper and environment settings
tests/                  Unit tests (use fakes; no model download)
docs/                   Reports, presentation, demo video
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit together.

## Project documents

| File | Description |
|---|---|
| [Problem statement](docs/report/01-problem-statement.pdf) | What the project set out to solve |
| [Abstract](docs/report/02-abstract.pdf) | Short project summary |
| [Full documentation](docs/report/03-project-documentation.pdf) | Mini-project report |
| [Review presentation](docs/presentation/review-presentation.pdf) | Slides from the project review |
| [Demo video](docs/media/demo.mp4) | Walkthrough of the original version |

The report and demo describe the original single-model version; this repository extends it.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md). Release notes are in
[CHANGELOG.md](CHANGELOG.md).

## Limitations

- A 248M-parameter model gives good-not-great summaries; use a larger model via `SUMMARIZER_MODEL`
  for better quality.
- Scanned PDFs have no text layer. Convert pages to images and use the OCR input.
- BM25 matches keywords, not meaning, so phrase questions with words from the document.

## License

[MIT](LICENSE) for the code. The default model is licensed **CC BY-NC 4.0 (non-commercial)**; see
[NOTICE.md](NOTICE.md) for third-party licenses.
