# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [1.0.0] - Unreleased

### Added
- `textsum` package with map-reduce summarization, BM25 retrieval, PDF Q&A and OCR.
- Streamlit app with Summarize, Ask your PDF and About tabs, length control and summary download.
- Unit tests, ruff configuration and a GitHub Actions workflow.
- `docs/` with the problem statement, abstract, report, presentation and demo video.
- README, architecture notes, contributing guide and `.env.example`.

### Changed
- Model is loaded from a Hugging Face id (`SUMMARIZER_MODEL`) instead of a hardcoded `D:\` path.
- Q&A no longer depends on LangChain/FAISS; the index is per session instead of a shared folder.

### Removed
- Hardcoded Google API key and committed FAISS files.
- Legacy `App.py`, `QA.py` and `Sumz.py`, and the commented-out dead code.

### Security
- The Google API key that was committed to the earlier repositories must be revoked; it remains in
  their git history.
