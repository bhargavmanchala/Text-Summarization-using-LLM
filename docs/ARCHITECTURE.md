# Architecture

## Overview

`app.py` is a thin Streamlit UI. All logic lives in the `textsum` package, which has no Streamlit
imports and can be reused or tested on its own.

```
Input (text / PDF / image)
   │  pdf_io.extract_text   ocr.image_to_text
   ▼
 plain text ──► Summarizer ──► summary
   │            (chunk → summarize → repeat)
   └──► DocumentQA ──► BM25Index ──► top passages ──► Gemini or local model ──► answer
```

## Summarization (`summarizer.py`)

The model has a limited input window, so the text is processed in rounds:

1. Encode the text to tokens and split into 400-token windows with 40 tokens of overlap
   (`chunking.chunk_sequence`).
2. Summarize each window with beam search (`no_repeat_ngram_size=3`).
3. Join the summaries. If more than one window was needed, repeat from step 1 on the joined text
   (at most 3 rounds). A single window ends the loop.

The length preset (`short`, `medium`, `long`) sets the per-window token limits.

## Question answering (`qa.py`, `retrieval.py`)

1. Split the document into ~180-word passages (40-word overlap).
2. Rank passages for the question with Okapi BM25 (pure Python, no vector database).
3. Build a prompt from the top 4 passages and ask Gemini if `GOOGLE_API_KEY` is set, otherwise the
   local model. The prompt tells the model to reply with a fixed "not found" sentence when the
   context lacks the answer.

The index lives in Streamlit's per-session state, so concurrent users do not share data.

## Design decisions

- **BM25 instead of embeddings + FAISS.** The original used Google embeddings and a shared FAISS
  index on disk. BM25 needs no API key, no extra dependencies and no files on disk, at the cost of
  keyword-only matching.
- **`TextModel` protocol.** `Summarizer` depends on `encode`/`decode`/`generate`, so tests use a
  tiny fake and never download weights.
- **Configuration through environment variables** (`config.py`); no secrets in source.

## Testing

`pytest` runs without ML libraries or network: chunking, BM25, config, summarizer logic (with a
fake model), Q&A flow and PDF extraction (including a real PDF from `docs/`).

## History

This repository consolidates four earlier repositories: `Text-Summarization-using-LLM`,
`Text-Summarization-using-LLMs`, `QA-Bot` and `Text-Summarization-using-LLM-Mini-Project-`.
